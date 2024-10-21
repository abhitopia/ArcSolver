#%%

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossCrossEntropy(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=None):
        """
        Focal Loss for Multi-Class Classification with Cross Entropy.

        Args:
            alpha (float): Weighting factor for the classes. Default is 1.0.
            gamma (float, optional): Focusing parameter to reduce the loss contribution from easy examples. Default is 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(FocalLossCrossEntropy, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Forward pass for focal loss.

        Args:
            inputs (torch.Tensor): Logits tensor of shape (N, C), where N is the batch size and C is the number of classes.
            targets (torch.Tensor): Ground truth labels of shape (N,) with integer values in [0, C-1].

        Returns:
            torch.Tensor: Computed focal loss.
        """
        # Compute the standard cross entropy loss
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            ignore_index=self.ignore_index
        )  # Shape: (N,)

        # Compute pt (probability of the true class)
        pt = torch.exp(-ce_loss)  # pt = exp(-loss) = P(class)

        # Compute the focal loss modulation factor
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Compute the focal loss
        focal_loss = focal_weight * ce_loss  # Shape: (N,)

        # Create a mask for valid targets (1 for valid, 0 for ignored)
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
        else:
            mask = torch.ones_like(focal_loss)

        # Apply the mask
        focal_loss = focal_loss * mask

        # Apply reduction
        if self.reduction == 'mean':
            # Avoid division by zero
            valid_count = mask.sum()
            if valid_count > 0:
                return focal_loss.sum() / valid_count
            else:
                return torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # Shape: (N,)


#%%
# Example Usage:

# Parameters
batch_size = 4
num_classes = 5
ignore_label = 2

# Initialize the loss function
# Example with class-wise alpha
alpha = 2.0  # Example class weights
criterion = FocalLossCrossEntropy(alpha=alpha, gamma=2.0, reduction='mean', ignore_index=ignore_label)

# Example logits and targets
torch.manual_seed(0)  # For reproducibility
logits = torch.randn(batch_size, num_classes, requires_grad=True)  # Shape: (N, C)
targets = torch.tensor([0, 1, ignore_label, 3])  # Shape: (N,)

# Compute loss
loss = criterion(logits, targets)
print(f"Focal Loss: {loss.item()}")

# Backward pass
loss.backward()

# %%
