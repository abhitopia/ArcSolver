#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%

class MultiLevelLoss(nn.Module):
    def __init__(self, PAD_IDX, pct_indices_per_level):
        super(MultiLevelLoss, self).__init__()
        self.PAD_IDX = PAD_IDX
        self.pct_indices_per_level = pct_indices_per_level

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

    def compute_loss_for_level(self, logits_i, targets, new_selected_mask, valid_mask):
        """
        Computes cross-entropy loss for the current level over newly selected positions.

        Args:
            logits_i (torch.Tensor): Logits tensor for the current level, shape (B, T, D).
            targets (torch.Tensor): Target tensor, shape (B, T).
            new_selected_mask (torch.Tensor): Mask of positions newly selected at this level, shape (B, T).
            valid_mask (torch.Tensor): Mask of valid positions, shape (B, T).

        Returns:
            loss_level (torch.Tensor): Loss value for the current level.
            num_tokens (int): Number of tokens used in the loss computation.
        """
        selected_positions = new_selected_mask & valid_mask  # Ensure only valid positions are considered

        num_tokens = selected_positions.sum().item()
        if num_tokens == 0:
            return 0.0, 0  # No positions to compute loss on

        # Extract logits and targets for selected positions
        logits_selected = logits_i[selected_positions]  # shape (Total_Selected_Positions, D)
        targets_selected = targets[selected_positions]  # shape (Total_Selected_Positions,)

        # Compute cross-entropy loss with 'sum' reduction
        loss_level = F.cross_entropy(logits_selected, targets_selected, reduction='sum')

        return loss_level, num_tokens

    def forward(self, logits_list, targets):
        """
        Computes the progressive loss over multiple levels.

        Args:
            logits_list (list of torch.Tensor): List of N logits tensors, each of shape (B, T, D).
            targets (torch.Tensor): Target tensor, shape (B, T).

        Returns:
            total_loss (torch.Tensor): The computed total loss.
        """
        B, T = targets.shape
        N = len(logits_list)  # Number of levels
        device = targets.device

        # Compute valid_mask and number of valid tokens per sequence
        valid_mask, num_valid_tokens_per_seq = self.compute_valid_mask(targets)

        # Initialize selected indices mask for each sequence
        selected_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        total_loss = 0.0
        total_tokens = 0

        for level_i in range(N):
            logits_i = logits_list[level_i]  # shape (B, T, D)
            pct_i = self.pct_indices_per_level[level_i]

            # Compute N_i for each sequence
            N_i_per_seq = (num_valid_tokens_per_seq.float() * pct_i).ceil().long()  # shape (B,)

            # Compute predictions and confidences
            preds, confidences = self.compute_predictions_and_confidences(logits_i)

            # Get correct predictions not already selected
            correct_mask = self.get_correct_mask(preds, targets, valid_mask, selected_mask)

            # Store a copy of selected_mask before updating
            previous_selected_mask = selected_mask.clone()

            # Update selected_mask with correct predictions
            selected_mask |= correct_mask

            # Select additional positions to meet N_i_per_seq
            selected_mask = self.select_additional_positions(
                confidences, valid_mask, selected_mask, N_i_per_seq
            )

            # Compute new selections made at this level
            new_selected_mask = selected_mask ^ previous_selected_mask

            # Compute loss for this level over newly selected positions
            loss_level, num_tokens_level = self.compute_loss_for_level(
                logits_i, targets, new_selected_mask, valid_mask
            )

            total_loss += loss_level
            total_tokens += num_tokens_level

        if total_tokens == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute the final loss by dividing the total loss by the total number of tokens
        final_loss = total_loss / total_tokens

        return final_loss


#%%

# Example tensors (replace with actual data)
B, T, D = 3, 5, 10  # Example dimensions
N_levels = 3
logits_list = [torch.randn(B, T, D) for _ in range(N_levels)]
targets = torch.randint(0, D, (B, T))
PAD_IDX = 0  # Index of the padding token
pct_indices_per_level = [0.5, 0.75, 1.0]  # Percentages for each level

# Initialize the ProgressiveLoss class
progressive_loss = MultiLevelLoss(PAD_IDX, pct_indices_per_level)

# Compute the loss
loss = progressive_loss(logits_list, targets)
print(f"Total Loss: {loss.item()}")


#%%
# Test compute_valid_mask
valid_mask, num_valid_tokens_per_seq = progressive_loss.compute_valid_mask(targets)
print("Valid Mask:", valid_mask)
print("Number of Valid Tokens per Sequence:", num_valid_tokens_per_seq)

# Test compute_predictions_and_confidences
logits_i = logits_list[0]
preds, confidences = progressive_loss.compute_predictions_and_confidences(logits_i)
print("Predictions:", preds)
print("Confidences:", confidences)

# Continue testing other methods similarly...


# %%
