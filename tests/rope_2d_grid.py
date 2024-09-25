#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory


import torch
import math
from src.rope import RotaryEmbedding, apply_rotary_emb

# Grid dimensions
height = 5
width = 4
grid_size = height * width

# Embedding dimensions
n_dim = 256  # Total embedding dimension
n_head = 4  # Number of attention heads
head_dim = n_dim // n_head

# Batch size
batch_size = 3

# Create identical embeddings for all tokens
embedding_vector = torch.randn(n_dim)  # Random vector for all tokens
x_grid = embedding_vector.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, height, width, n_dim)

x_grid.shape
# %%
# Reshape x_grid for multi-head attention [B, n_head, H, W, head_dim]
x_grid = x_grid.reshape(batch_size, height, width, n_head, head_dim).permute(0, 3, 1, 2, 4)

# Use the same embeddings for queries and keys
q = x_grid
k = x_grid
q.shape, k.shape
# %%
# Initialize the rotary embedding for pixel data
pos_emb = RotaryEmbedding(
    dim=head_dim // 2,
    freqs_for='pixel',
    max_freq=max(height, width)
)

# Compute axial frequencies
freqs = pos_emb.get_axial_freqs(height, width)  # Shape: [H, W, HEAD_DIM]

# Expand freqs to match batch size and number of heads
freqs = freqs.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W, HEAD_DIM]

# Apply rotary embeddings to queries and keys
q_rotated = apply_rotary_emb(freqs, q)
k_rotated = apply_rotary_emb(freqs, k)

# %%
q_rotated.shape, k_rotated.shape
# %%
# Reshape to [B, N_HEAD, SEQ_LEN, HEAD_DIM]
q_rotated = q_rotated.reshape(batch_size, n_head, grid_size, head_dim)
k_rotated = k_rotated.reshape(batch_size, n_head, grid_size, head_dim)
q_rotated.shape, k_rotated.shape
# %%
def position_to_index(row, col, width):
    return row * width + col

# Pair 1 indices (Positions (0,0) and (1,1))
idx_A = position_to_index(2, 2, width)
idx_B = position_to_index(0, 0, width)

# Pair 2 indices (Positions (2,1) and (3,2))
idx_C = position_to_index(2, 1, width)
idx_D = position_to_index(4, 3, width)

q_A = q_rotated[:, :, idx_A, :]
k_B = k_rotated[:, :, idx_B, :]
q_C = q_rotated[:, :, idx_C, :]
k_D = k_rotated[:, :, idx_D, :]

# Compute attention scores for each pair
score_AB = torch.einsum('bhc,bhc->bh', q_A, k_B) / math.sqrt(head_dim)
score_CD = torch.einsum('bhc,bhc->bh', q_C, k_D) / math.sqrt(head_dim)
# %%
print("Attention scores between A and B:")
print(score_AB)

print("\nAttention scores between C and D:")
print(score_CD)

assert torch.allclose(score_AB, score_CD), "Attention scores between different pairs do not match."
# %%
