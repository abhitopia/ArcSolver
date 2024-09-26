import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast


import torch
import torch.nn as nn
import math

class Rope2D(nn.Module):
    def __init__(self, h_dim, max_height, max_width, device='cpu'):
        """
        Initializes the Rope2D module.

        Args:
            h_dim (int): Head dimension (must be divisible by 2).
            max_height (int): Maximum height for positional indices.
            max_width (int): Maximum width for positional indices.
            device (str): Device to store buffers ('cpu' or 'cuda').
        """
        super().__init__()
        self.head_dim = h_dim
        self.max_height = max_height
        self.max_width = max_width
        self.max_freq = max(max_height, max_width)
        self.device = device

        assert h_dim % 2 == 0, 'The head dimension must be divisible by 2.'
        self.dim = h_dim // 2  # Dimension per axis

        # Compute axial frequencies: [max_height, max_width, h_dim]
        max_freqs = self.get_axial_freqs_pytorch(max_height, max_width)  # [height, width, h_dim]

        # Precompute cosine and sine frequencies (In high precision)
        cos_freqs = torch.cos(max_freqs).float()  # [height, width, h_dim]
        sin_freqs = torch.sin(max_freqs).float()  # [height, width, h_dim]

        # Register cosine and sine frequencies as buffers
        self.register_buffer('cos_freqs', cos_freqs, persistent=False)
        self.register_buffer('sin_freqs', sin_freqs, persistent=False)

    @autocast(enabled=False)
    def get_axial_freqs_pytorch(self, *dims):
        """
        Computes axial frequencies for given dimensions without using `einops`.

        Args:
            *dims (int): Dimensions for each axis (e.g., height, width).

        Returns:
            Tensor: Combined axial frequencies of shape [dim1, dim2, ..., h_dim].
        """
        all_freqs = []

        # Initialize frequency buffer: [dim//2], Unsure full precision
        freqs = (torch.linspace(1., self.max_freq / 2, self.dim // 2, device=self.device) * math.pi).float()  # [dim//2]
        for ind, dim_size in enumerate(dims):
            # Generate positional values, full precision
            pos = torch.linspace(-1., 1., steps=dim_size, device=self.device).float()  # [dim_size]

            # Compute outer product: [dim_size, dim//2]
            freqs_dim = torch.einsum('i,j->ij', pos.type(freqs.dtype), freqs)  # [dim_size, dim//2]

            # Repeat each frequency pair (sine and cosine)
            freqs_dim = freqs_dim.repeat_interleave(2, dim=-1)  # [dim_size, h_dim]

            # Reshape freqs to align with the current axis
            expand_dims = [1] * len(dims)  # e.g., for 2D: [1, 1]
            expand_dims[ind] = dim_size  # Set the current axis dimension

            freqs_dim = freqs_dim.view(*expand_dims, self.dim)  # [dim_size,1,h_dim] or [1,dim_size,h_dim]

            all_freqs.append(freqs_dim)

        # Broadcast all frequencies to have the same shape
        all_freqs_broadcasted = torch.broadcast_tensors(*all_freqs)  # List of tensors with shape [dim1, dim2, ..., h_dim]

        # Combine frequencies by summing them
        combined_freqs = torch.cat(all_freqs_broadcasted, dim=-1)  # [dim1, dim2, ..., rot_dim * number_of_axes]

        return combined_freqs  # [dim1, dim2, ..., h_dim]

    @staticmethod
    def rotate_half(x):
        """
        Rotates the last two dimensions of the input tensor.

        Args:
            x (Tensor): Tensor of shape [..., rot_dim].

        Returns:
            Tensor: Rotated tensor of the same shape.
        """
        # Ensure rot_dim is even
        assert x.shape[-1] % 2 == 0, "rot_dim must be even for rotation."

        # Reshape to separate pairs: [..., rot_dim//2, 2]
        x = x.view(*x.shape[:-1], -1, 2)

        # Unbind the last dimension
        x1, x2 = x.unbind(-1)

        # Stack with rotation: [-x2, x1]
        x_rotated = torch.stack((-x2, x1), dim=-1)

        # Flatten back to original shape: [..., rot_dim]
        return x_rotated.flatten(-2)

    @staticmethod
    @autocast( enabled=False)
    def apply_rotary_emb(x, cos, sin):
        """
        Applies rotary embeddings to the input tensor.

        Args:
            x (Tensor): [batch_size, n_head, S, h_dim].
            cos (Tensor): [batch_size, n_head, S, h_dim].
            sin (Tensor): [batch_size, n_head, S, h_dim].

        Returns:
            Tensor: Rotated tensor of shape [batch_size, n_head, S, h_dim].
        """
        x_fp = x.float() # ensure full precision computation
        # Apply rotary embeddings
        rotated_x_fp = Rope2D.rotate_half(x_fp)  # Rotate the first half
        x_rotated_fp = (x_fp * cos) + (rotated_x_fp * sin)
        return x_rotated_fp.type(x.dtype) # Cast back if needed

    def forward(self, x, x_indices):
        """
        Applies 2D Rotary Positional Embeddings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [batch_size, n_head, S, h_dim].
            x_indices (Tensor): Positional indices of shape [batch_size, S, 2],
                                where each position is (row, col).

        Returns:
            Tensor: Tensor with rotary embeddings applied, shape [batch_size, n_head, S, h_dim].
        """
        # batch_size, n_head, S, h_dim = x.shape
        # device = x.device

        # Extract row and column indices
        row_indices = x_indices[..., 0]  # [batch_size, S]
        col_indices = x_indices[..., 1]  # [batch_size, S]

        # Create a mask for valid positions (assuming negative indices are invalid)
        valid_mask = (row_indices >= 0) & (col_indices >= 0)  # [batch_size, S]

        # Replace invalid indices with zeros to prevent indexing errors
        row_indices = torch.where(valid_mask, row_indices, torch.zeros_like(row_indices))
        col_indices = torch.where(valid_mask, col_indices, torch.zeros_like(col_indices))

        # Fetch cosine and sine frequencies based on positions
        # [batch_size, S, h_dim]
        cos_freqs = self.cos_freqs[row_indices.long(), col_indices.long()]
        sin_freqs = self.sin_freqs[row_indices.long(), col_indices.long()]

        # Reshape to match x's dimensions
        # [batch_size, 1, S, h_dim]
        cos_freqs = cos_freqs.unsqueeze(1)
        sin_freqs = sin_freqs.unsqueeze(1)

        # Apply rotary embeddings
        with torch.cuda.amp.autocast(enabled=False):
            x_rotated = self.apply_rotary_emb(x, cos_freqs, sin_freqs)  # [batch_size, n_head, S, h_dim]

        # For invalid positions, retain the original x without rotation
        if not valid_mask.all():
            # Expand mask to match x's dimensions
            # [batch_size, n_head, S, h_dim]
            valid_mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).expand(-1, n_head, -1, h_dim)

            # Use torch.where to select rotated or original x based on the mask
            x_rotated = torch.where(valid_mask_expanded, x_rotated, x)

        return x_rotated  # [batch_size, n_head, S, h_dim]
