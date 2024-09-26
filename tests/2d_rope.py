#%%

import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory

import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import math
from einops import rearrange

from src.rope import RotaryEmbedding


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

        # Initialize frequency buffer: [dim//2]
        freqs = torch.linspace(1., self.max_freq / 2, self.dim // 2, device=device) * math.pi  # [dim//2]
        self.register_buffer('freqs_buffer', freqs, persistent=True)

        # Compute axial frequencies: [max_height, max_width, h_dim]
        max_freqs = self.get_axial_freqs_pytorch(max_height, max_width)  # [height, width, h_dim]
        print("Print Freq", max_freqs.shape)

        # Precompute cosine and sine frequencies
        cos_freqs = torch.cos(max_freqs)  # [height, width, h_dim]
        sin_freqs = torch.sin(max_freqs)  # [height, width, h_dim]

        # Register cosine and sine frequencies as buffers
        self.register_buffer('cos_freqs', cos_freqs, persistent=False)
        self.register_buffer('sin_freqs', sin_freqs, persistent=False)

    def get_axial_freqs_pytorch(self, *dims):
        """
        Computes axial frequencies for given dimensions without using `einops`.

        Args:
            *dims (int): Dimensions for each axis (e.g., height, width).

        Returns:
            Tensor: Combined axial frequencies of shape [dim1, dim2, ..., h_dim].
        """
        all_freqs = []
        for ind, dim_size in enumerate(dims):
            # Generate positional values
            pos = torch.linspace(-1., 1., steps=dim_size, device=self.device)  # [dim_size]

            # Compute outer product: [dim_size, dim//2]
            freqs = torch.einsum('i,j->ij', pos.type(self.freqs_buffer.dtype), self.freqs_buffer)  # [dim_size, dim//2]

            # Repeat each frequency pair (sine and cosine)
            freqs = freqs.repeat_interleave(2, dim=-1)  # [dim_size, h_dim]

            # Reshape freqs to align with the current axis
            expand_dims = [1] * len(dims)  # e.g., for 2D: [1, 1]
            expand_dims[ind] = dim_size  # Set the current axis dimension

            freqs = freqs.view(*expand_dims, self.dim)  # [dim_size,1,h_dim] or [1,dim_size,h_dim]

            all_freqs.append(freqs)

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
        # Apply rotary embeddings
        rotated_x = Rope2D.rotate_half(x)  # Rotate the first half
        print(x.shape, rotated_x.shape, cos.shape, sin.shape)
        x_rotated = (x * cos) + (rotated_x * sin)
        return x_rotated.type(x.dtype)

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
        batch_size, n_head, S, h_dim = x.shape
        device = x.device

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
        x_rotated = self.apply_rotary_emb(x, cos_freqs, sin_freqs)  # [batch_size, n_head, S, h_dim]

        # For invalid positions, retain the original x without rotation
        if not valid_mask.all():
            # Expand mask to match x's dimensions
            # [batch_size, n_head, S, h_dim]
            valid_mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).expand(-1, n_head, -1, h_dim)

            # Use torch.where to select rotated or original x based on the mask
            x_rotated = torch.where(valid_mask_expanded, x_rotated, x)

        return x_rotated  # [batch_size, n_head, S, h_dim]



#%%
import torch
import torch.nn as nn
import math

# Parameters
batch_size = 1  # For simplicity
n_dim = 256
n_head = 4
head_dim = n_dim // n_head
seq_len = 4  # For testing
max_height = 5
max_width = 5


# Reshape x_grid for multi-head attention [B, n_head, H, W, head_dim]
embedding_vector = torch.randn(n_dim)  # Random vector for all tokens
x_grid = embedding_vector.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, max_height, max_width, n_dim)
x_grid = x_grid.reshape(batch_size, max_height, max_width, n_head, head_dim).permute(0, 3, 1, 2, 4)

x_grid.shape

# Mock data
# torch.manual_seed(0)  # For reproducibility
# q = torch.randn(batch_size, n_head, seq_len, head_dim)
# k = torch.randn(batch_size, n_head, seq_len, head_dim)

q = x_grid
k = x_grid

# Define per-token positions
positions = torch.tensor([
    [[1, 1], [3, 3], [2, 2], [4, 4]],  # Batch 0
], dtype=torch.long)  # Shape: [batch_size, seq_len, 2]

# Initialize RoPE2D module
rope2d = Rope2D(h_dim=head_dim, max_height=max_height, max_width=max_width)


positions.shape, q.shape, k.shape
#%%
# Apply RoPE2D
q_rotated = rope2d(q, positions)
k_rotated = rope2d(k, positions)

# Compute attention scores between pairs
def compute_attention_score(q, k):
    # q, k: [batch_size, n_head, head_dim]
    score = torch.einsum('b n d, b n d -> b n', q, k) / math.sqrt(head_dim)
    return score  # [batch_size, n_head]

# Indices for tokens with same relative positions
idx_A = 0  # Token at (1,1)
idx_B = 1  # Token at (3,3)
idx_C = 2  # Token at (2,2)
idx_D = 3  # Token at (4,4)

# Extract queries and keys
q_A = q_rotated[:, :, idx_A, :]
k_B = k_rotated[:, :, idx_B, :]
q_C = q_rotated[:, :, idx_C, :]
k_D = k_rotated[:, :, idx_D, :]

# Compute attention scores
score_AB = compute_attention_score(q_A, k_B)
score_CD = compute_attention_score(q_C, k_D)

print("Attention scores between A and B:")
print(score_AB)

print("\nAttention scores between C and D:")
print(score_CD)

# Verify if the scores are approximately equal
if torch.allclose(score_AB, score_CD, atol=1e-6):
    print("\nAttention scores are the same for pairs with the same relative positions.")
else:
    print("\nAttention scores differ for pairs with the same relative positions.")

# %%
