#%%

import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory

from src.rope_2d import Rope2D
import torch
import torch.nn as nn
import math

#%%

# Parameters
batch_size = 1  # For simplicity
n_dim = 256
n_head = 4
head_dim = n_dim // n_head
seq_len = 4  # For testing
max_height = 30
max_width = 30


# Reshape x_grid for multi-head attention [B, n_head, H, W, head_dim]
embedding_vector = torch.randn(n_dim)  # Random vector for all tokens
# x_grid = embedding_vector.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, max_height, max_width, n_dim)
# print("X Grid Shape",x_grid.shape)
# x_grid = x_grid.reshape(batch_size, max_height, max_width, n_head, head_dim).permute(0, 3, 1, 2, 4)
# print("X Grid Shape",x_grid.shape)

# # Mock data
# torch.manual_seed(0)  # For reproducibility
# q = torch.randn(batch_size, n_head, seq_len, head_dim)
# k = torch.randn(batch_size, n_head, seq_len, head_dim)

# q = x_grid
# k = x_grid

# Define per-token positions
positions = torch.tensor([
    [[1, 1], [4, 3], [2, 2], [8, 6]],  # Batch 0
], dtype=torch.long)  # Shape: [batch_size, seq_len, 2]

# Initialize RoPE2D module
rope2d = Rope2D(h_dim=head_dim, max_height=max_height, max_width=max_width)


q = embedding_vector.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, n_dim)
q = q.reshape(batch_size, seq_len, n_head, head_dim).permute(0, 2, 1, 3)

positions.shape, q.shape
#%%
# Apply RoPE2D
q_rotated = rope2d(q, positions)
q_rotated
#%%
# Compute attention scores between pairs
def compute_attention_score(q, k):
    # q, k: [batch_size, n_head, head_dim]
    score = torch.einsum('b n d, b n d -> b n', q, k) / math.sqrt(head_dim)
    return score  # [batch_size, n_head]

# Indices for tokens with same relative positions
idx_A = 0  # Token at (1,1)
idx_B = 1  # Token at (4,3)
idx_C = 2  # Token at (2,2)
idx_D = 3  # Token at (5,4)

# Extract queries and keys
q_A = q_rotated[:, :, idx_A, :]
k_B = q_rotated[:, :, idx_B, :]
q_C = q_rotated[:, :, idx_C, :]
k_D = q_rotated[:, :, idx_D, :]

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