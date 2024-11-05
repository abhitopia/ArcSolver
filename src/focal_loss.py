#%%
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F



def focal_cross_entropy(outputs, targets, gamma: float = 2.0, label_smoothing: float = 0.0,  alpha: float = 1.0, reduction: str = 'mean', ignore_index: Optional[int] = None):
    """
    Args:
        outputs (torch.Tensor): Logits tensor of shape (B, S, D).
        targets (torch.Tensor): Ground truth labels of shape (B, S) with integer values in [0, D-1].
        alpha (float, optional): Weighting factor for the target class. Default is 1.0.
        gamma (float, optional): Focusing parameter to reduce the loss contribution from easy examples. Default is 2.0.
        label_smoothing (float, optional): Label smoothing factor that smooths the one-hot encoded target distribution. Default is 0.0. (0.1 means 90% to the true class and 10% spread to the other classes)
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.
        ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
    Returns:
        torch.Tensor: Computed focal loss.
    """
    # Ensure outputs and targets are of expected dimensions
    if outputs.dim() != 3:
        raise ValueError(f"Expected outputs of shape (B, S, D), got {outputs.shape}")
    if targets.dim() != 2:
        raise ValueError(f"Expected targets of shape (B, S), got {targets.shape}")

    B, S, D = outputs.shape

    # Flatten the batch and sequence dimensions
    outputs = outputs.view(-1, D)  # Shape: (B*S, D)
    targets = targets.view(-1)     # Shape: (B*S)

    # Handle ignore_index if specified
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        outputs = outputs[valid_mask]
        targets = targets[valid_mask]

    # Compute the standard cross entropy loss without reduction
    ce_loss = F.cross_entropy(
        outputs,
        targets,
        label_smoothing=label_smoothing,
        reduction='none'
    )  # Shape: (N,)

    # Compute pt (probability of the true class)
    pt = torch.exp(-ce_loss)  # pt = exp(-loss) = P(class)

    # Compute the focal loss modulation factor
    focal_weight = alpha * (1 - pt) ** gamma

    # Compute the focal loss
    focal_loss = focal_weight * ce_loss  # Shape: (N,)

    # Apply reduction
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss  # Shape: (N,)



def focal_bce(outputs, targets, gamma: float = 2.0, alpha: float = 1.0, reduction: str = 'mean', ignore_index: Optional[int] = None):
        """
        Args:
            outputs (torch.Tensor): Logits tensor of shape (B, S, D).
            targets (torch.Tensor): Ground truth labels of shape (B, S) with integer values in [0, D-1].
            alpha (float, optional): Weighting factor for the target class. Default is 1.0.
            gamma (float, optional): Focusing parameter to reduce the loss contribution from easy examples. Default is 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Ensure outputs and targets are of expected dimensions
        if outputs.dim() != 3:
            raise ValueError(f"Expected outputs of shape (B, S, D), got {outputs.shape}")
        if targets.dim() != 2:
            raise ValueError(f"Expected targets of shape (B, S), got {targets.shape}")

        B, S, D = outputs.shape

        # Flatten the batch and sequence dimensions
        outputs = outputs.view(-1, D)  # Shape: (B*S, D)
        targets = targets.view(-1)      # Shape: (B*S)

        # Handle ignore_index if specified
        if ignore_index is not None:
            valid_mask = targets != ignore_index
            outputs = outputs[valid_mask]
            targets = targets[valid_mask]

        # Select the logits corresponding to the target classes
        # Shape of targets.unsqueeze(1): (N, 1)
        # Shape of logits_target: (N,)
        logits_target = outputs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # # Compute the probabilities for the target class using sigmoid
        # probs = torch.sigmoid(logits_target)


        # Compute the binary cross-entropy loss with logits for the target class
        # Since we're only focusing on the target class, the target is 1
        BCE_loss = F.binary_cross_entropy_with_logits(logits_target, torch.ones_like(logits_target), reduction='none')


        # Compute the probabilities for the target class
        probs = torch.exp(-BCE_loss)

        # Compute the focal loss modulation factor
        focal_weight = alpha * (1 - probs) ** gamma

        # Compute the focal loss
        F_loss = focal_weight * BCE_loss

        # Apply reduction
        if reduction == 'mean':
            return F_loss.mean()
        elif reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss  # Shape: (N,)

#%%
# # Example Usage:
# # Suppose you have 10 classes
# D = 10
# criterion = FocalLossMultiClassSelective(alpha=1.0, gamma=2.0, reduction='mean')

# # Example tensors
# B, S = 32, 20  # Batch size and sequence length
# outputs = torch.randn(B, S, D, requires_grad=True)  # Logits tensor
# targets = torch.randint(0, D, (B, S))               # Targets tensor

# # Compute loss
# loss = criterion(outputs, targets)
# print(f"Focal Loss: {loss.item()}")

# # Backward pass
# loss.backward()
# %%
