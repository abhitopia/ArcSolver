#%%
from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def exp_spacing(
    n: int,
    rate: float = 0.0,
    min_val: float = 0.4,
    max_val: float = 1.0,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generates n points in the range [min_val, max_val] with spacing controlled by rate.

    Parameters:
    - n (int): Number of points to generate. Must be >= 1.
    - rate (float): Controls the spacing between points.
        - rate = 0: Uniform spacing.
        - rate = 1: All points except the first are max_val.
        - rate = -1: All points except the last are min_val.
        - -1 < rate < 1: Exponential spacing biased towards min_val (rate > 0) or max_val (rate < 0).
    - min_val (float): Minimum value of the range.
    - max_val (float): Maximum value of the range.
    - device (torch.device): Device on which to perform computations.
    - dtype (torch.dtype): Data type of the output tensor.

    Returns:
    - torch.Tensor: Tensor of n points with specified spacing.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if n == 1:
        return torch.tensor([min_val], device=device, dtype=dtype)
    if n == 2:
        return torch.tensor([min_val, max_val], device=device, dtype=dtype)
    
    # Define a fixed scale for exponential mapping
    scale = 10.0  # Adjust as needed for desired bias strength
    
    # Handle extreme rates: rate=1 and rate=-1
    if rate >= 1.0:
        # All points except the first are max_val
        points = torch.full((n,), max_val, device=device, dtype=dtype)
        points[0] = min_val
        return points
    elif rate <= -1.0:
        # All points except the last are min_val
        points = torch.full((n,), min_val, device=device, dtype=dtype)
        points[-1] = max_val
        return points
    
    # Generate uniform parameter t
    t = torch.linspace(0, 1, steps=n, device=device, dtype=dtype)
    
    if rate == 0.0:
        p = t
    else:
        # Convert rate to tensor
        rate_tensor = torch.tensor(rate, device=device, dtype=dtype)
        
        # Compute exponential scaling
        exponent = scale * rate_tensor
        # For positive rate: p(t) = (1 - exp(-exponent * t)) / (1 - exp(-exponent))
        # For negative rate: p(t) = (exp(exponent * t) - 1) / (exp(exponent) - 1)
        if rate > 0:
            numerator = 1 - torch.exp(-exponent * t)
            denominator = 1 - torch.exp(-exponent)
        else:
            exponent = -exponent  # Make exponent positive for rate < 0
            numerator = torch.exp(exponent * t) - 1
            denominator = torch.exp(exponent) - 1
        
        p = numerator / denominator
        
    # Map p from [0, 1] to [min_val, max_val]
    points = min_val + p * (max_val - min_val)
    
    # Ensure points are within [min_val, max_val]
    points = torch.clamp(points, min=min_val, max=max_val)
    
    return points
#%%


class MultiLevelLoss(nn.Module):
    def __init__(self, pad_idx: int, edr: float = 2, min_pct: float = 0.4, max_pct: float = 1.0):
        super(MultiLevelLoss, self).__init__()
        self.PAD_IDX = pad_idx
        self.edr = edr
        self.min_pct = min_pct
        self.max_pct = max_pct

    def compute_valid_mask(self, targets):
        """
        Computes a mask identifying valid (non-padding) positions in the targets tensor.

        Args:
            targets (torch.Tensor): Target tensor of shape (B, T).

        Returns:
            valid_mask (torch.Tensor): Boolean mask of shape (B, T).
            num_valid_tokens_per_seq (torch.Tensor): Number of valid tokens per sequence, shape (B,).
        """
        valid_mask = (targets != self.PAD_IDX)  # shape (B, T)
        num_valid_tokens_per_seq = valid_mask.sum(dim=1)  # shape (B,)
        return valid_mask, num_valid_tokens_per_seq

    def compute_predictions_and_confidences(self, logits):
        """
        Computes predictions and confidence scores from logits.

        Args:
            logits (torch.Tensor): Logits tensor of shape (B, T, D).

        Returns:
            preds (torch.Tensor): Predicted token indices, shape (B, T).
            confidences (torch.Tensor): Confidence scores for predictions, shape (B, T).
        """
        preds = logits.argmax(dim=2)  # shape (B, T)
        log_probs = F.log_softmax(logits, dim=2)  # shape (B, T, D)
        confidences = log_probs.gather(2, preds.unsqueeze(2)).squeeze(2)  # shape (B, T)
        return preds, confidences

    def get_correct_mask(self, preds, targets, valid_mask, selected_mask):
        """
        Identifies correctly predicted positions not already selected.

        Args:
            preds (torch.Tensor): Predicted token indices, shape (B, T).
            targets (torch.Tensor): Target tensor, shape (B, T).
            valid_mask (torch.Tensor): Mask of valid positions, shape (B, T).
            selected_mask (torch.Tensor): Mask of positions already selected, shape (B, T).

        Returns:
            correct_mask (torch.Tensor): Mask of correct predictions to select, shape (B, T).
        """
        correct_mask = (preds == targets) & valid_mask & (~selected_mask)  # shape (B, T)
        return correct_mask

    def select_additional_positions(self, confidences, valid_mask, selected_mask, N_i_per_seq):
        """
        Selects additional positions based on the highest confidence scores.

        Args:
            confidences (torch.Tensor): Confidence scores, shape (B, T).
            valid_mask (torch.Tensor): Mask of valid positions, shape (B, T).
            selected_mask (torch.Tensor): Mask of positions already selected, shape (B, T).
            N_i_per_seq (torch.Tensor): Number of positions to select per sequence, shape (B,).

        Returns:
            selected_mask (torch.Tensor): Updated mask with additional positions selected, shape (B, T).
        """
        B, T = confidences.shape
        device = confidences.device

        # Calculate how many more positions we need to select per sequence
        num_selected_per_seq = (selected_mask & valid_mask).sum(dim=1)  # shape (B,)
        need_more = N_i_per_seq - num_selected_per_seq  # shape (B,)
        need_more = torch.clamp(need_more, min=0)

        # Remaining positions not yet selected and are valid
        remaining_mask = valid_mask & (~selected_mask)  # shape (B, T)
        num_remaining_per_seq = remaining_mask.sum(dim=1)  # shape (B,)

        # Limit need_more to the number of remaining positions
        select_lengths = torch.min(need_more, num_remaining_per_seq)  # shape (B,)

        # Prepare confidences for remaining positions
        confidences_remaining = confidences.clone()  # shape (B, T)
        confidences_remaining[~remaining_mask] = -float('inf')  # Assign -inf to positions not in remaining_mask

        # Sort confidences_remaining along T dimension in descending order
        sorted_confidences, sorted_indices = confidences_remaining.sort(dim=1, descending=True)  # shape (B, T)

        # Create selection mask based on select_lengths
        idxs = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # Shape (B, T)
        select_mask = idxs < select_lengths.unsqueeze(1)  # Shape (B, T)

        # Flatten batch_indices, sorted_indices, select_mask
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, T)  # Shape (B, T)
        batch_indices_flat = batch_indices.reshape(-1)  # Shape (B*T,)
        sorted_indices_flat = sorted_indices.reshape(-1)  # Shape (B*T,)
        select_mask_flat = select_mask.reshape(-1)  # Shape (B*T,)

        # Get selected indices
        selected_batch_indices = batch_indices_flat[select_mask_flat]  # Shape (total_selected_positions,)
        selected_position_indices = sorted_indices_flat[select_mask_flat]  # Shape (total_selected_positions,)

        # Update selected_mask with new selections
        selected_mask[selected_batch_indices, selected_position_indices] = True

        return selected_mask
    

    def compute_loss_for_level(
            self,
            logits_i: torch.Tensor,
            targets: torch.Tensor,
            selected_mask: torch.Tensor,
            valid_mask: torch.Tensor
            ) -> Tuple[torch.Tensor, int]:
        """
        Computes cross-entropy loss for the current level over selected positions.

        Args:
            logits_i (torch.Tensor): Logits tensor for the current level, shape (B, T, D).
            targets (torch.Tensor): Target tensor, shape (B, T).
            selected_mask (torch.Tensor): Mask of positions selected up to this level, shape (B, T).
            valid_mask (torch.Tensor): Mask of valid positions, shape (B, T).

        Returns:
            Tuple[torch.Tensor, int]: Loss value for the current level and the number of tokens used in the loss computation at this level.
        """
        # Ensure only valid positions are considered
        selected_positions: torch.Tensor = selected_mask & valid_mask

        num_tokens_level = int(selected_positions.sum().item())
        if num_tokens_level == 0:
            loss_level = torch.tensor(0.0, device=logits_i.device)
            return loss_level, 0

        # Extract logits and targets for selected positions
        logits_selected = logits_i[selected_positions]  # Shape: (Total_Selected_Positions, D)
        targets_selected = targets[selected_positions]  # Shape: (Total_Selected_Positions,)

        # Compute cross-entropy loss with 'sum' reduction
        loss_level = F.cross_entropy(logits_selected, targets_selected, reduction='sum')

        return loss_level, num_tokens_level
    

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the progressive loss over multiple levels.

        Args:
            logits (torch.Tensor): logits tensors combined into a single (N, B, T, D).
            targets (torch.Tensor): Target tensor, shape (B, T).

        Returns:
            total_loss (torch.Tensor): The computed total loss.
        """
        B, T = targets.shape
        N = logits.size(0)  # Number of levels
        device = targets.device

        pct_indices_per_level = exp_spacing(N, self.edr, self.min_pct, self.max_pct)
        # Compute valid_mask and number of valid tokens per sequence
        valid_mask, num_valid_tokens_per_seq = self.compute_valid_mask(targets)

        # Initialize selected indices mask for each sequence
        selected_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        # total_loss = 0.0
        total_loss: torch.Tensor = torch.tensor(0.0, device=device)

        total_tokens = 0

        for level_i in range(N):
            logits_i = logits[level_i]  # shape (B, T, D)
            pct_i = pct_indices_per_level[level_i]

            if pct_i == 0:
                continue

            # Compute N_i for each sequence
            N_i_per_seq = (num_valid_tokens_per_seq.float() * pct_i).ceil().long()  # shape (B,)

            # Compute predictions and confidences
            preds, confidences = self.compute_predictions_and_confidences(logits_i)

            # Get correct predictions not already selected
            correct_mask = self.get_correct_mask(preds, targets, valid_mask, selected_mask)

            # Update selected_mask with correct predictions
            selected_mask |= correct_mask

            # Select additional positions to meet N_i_per_seq
            selected_mask = self.select_additional_positions(
                confidences, valid_mask, selected_mask, N_i_per_seq
            )

            # Compute loss for this level over all selected positions
            loss_level, num_tokens_level = self.compute_loss_for_level(
                logits_i, targets, selected_mask, valid_mask
            )

            total_loss += loss_level
            total_tokens += num_tokens_level  # Accumulate total tokens over levels

        if total_tokens == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        final_loss = total_loss / total_tokens

        return final_loss



# #%%

# # Example tensors (replace with actual data)
# B, T, D = 3, 20, 1000  # Example dimensions
# N_levels = 4
# vocab_size = D
# logits_list = [torch.randn(B, T, D) for _ in range(N_levels)]
# targets = torch.randint(1, vocab_size, (B, T))
# PAD_IDX = 0  # Index of the padding token
# pct_indices_per_level = [0.5, 0.75, 1.0]  # Percentages for each level

# for i in range(B):
#     seq_len_i = torch.randint(T//2, T, (1,))[0].item()
#     targets[i, seq_len_i:] = 0  # Add padding tokens to targets

# targets

# progressive_loss = MultiLevelLoss(PAD_IDX, edr=-1, min_pct=0.0, max_pct=1.0)

# loss = progressive_loss(torch.stack(logits_list, dim=0), targets)
# print(f"Total Loss: {loss.item()}")
# #%%
# # Initialize the ProgressiveLoss class

# valid_mask, num_valid_tokens_per_seq = progressive_loss.compute_valid_mask(targets)
# print("Valid Mask:", valid_mask)
# print("Number of Valid Tokens per Sequence:", num_valid_tokens_per_seq)
# #%%
# # Test compute_valid_mask


# # Test compute_predictions_and_confidences
# logits_i = logits_list[1]
# preds, confidences = progressive_loss.compute_predictions_and_confidences(logits_i)
# print("Predictions:", preds)
# print("Confidences:", confidences)

# bid  = 2
# tid = 10

# # F.log_softmax(logits_i[test_bid, test_tid, :], dim=0)
# logits_i[bid, tid, :].unsqueeze(0).shape
# confid = F.log_softmax(logits_i[bid, tid, :].unsqueeze(0), dim=1).max()
# predid = logits_i[bid, tid, :].argmax()
# confid, confidences[bid, tid], predid, preds[bid, tid]
# #%%

# selected_mask = torch.zeros(B, T, dtype=torch.bool)
# # selected_mask[1, 2] = True
# correct_mask = progressive_loss.get_correct_mask(preds, targets, valid_mask, selected_mask)
# correct_mask

# #%%
# targets
# preds
# valid_mask
# #%%
# # Continue testing other methods similarly...

# # Compute the loss

# # %%

# # %%

# %%
