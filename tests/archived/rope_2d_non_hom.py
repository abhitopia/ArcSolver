#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory


import torch
import math
from src.archived.rope import RotaryEmbedding, apply_rotary_emb

#%%

def get_item_freqs(pos_emb, height, width, head_dim):
    # Compute axial frequencies for the item's height and width
    freqs = pos_emb.get_axial_freqs(height, width)  # Shape: [height, width, head_dim]
    # Flatten frequencies to match the sequence
    freqs = freqs.reshape(height * width, head_dim)  # Shape: [seq_len, head_dim]
    return freqs


def pad_freqs(freqs, max_seq_len):
    seq_len, head_dim = freqs.shape
    padding_len = max_seq_len - seq_len
    padding = torch.zeros(padding_len, head_dim, device=freqs.device, dtype=freqs.dtype)
    freqs_padded = torch.cat([freqs, padding], dim=0)
    return freqs_padded  # Shape: [max_seq_len, head_dim]

def get_batched_freqs(pos_emb, batch_heights, batch_widths, max_seq_len, head_dim):
    batch_size = len(batch_heights)
    # rot_dim = pos_emb.dim * pos_emb.number_of_axes  # rot_dim from your pos_emb
    batched_freqs = torch.zeros(batch_size, max_seq_len, head_dim, device=pos_emb.freqs.device)
    max_height = max(batch_heights)
    max_width = max(batch_widths)
    for i in range(batch_size):
        height = batch_heights[i]
        width = batch_widths[i]
        freqs = get_item_freqs(pos_emb, height, width, head_dim)
        freqs_padded = pad_freqs(freqs, max_seq_len) # Shape: [max_seq_len, head_dim]
        batched_freqs[i] = freqs_padded

    return batched_freqs.reshape(batch_size, max_height, max_width, head_dim)  # Shape: [batch_size, max_seq_len, head_dim]


# Assume q and k are of shape [B, N_HEADS, S, HEAD_DIM]
def apply_batched_rotary_emb(freqs, q, k):
    freqs = freqs.unsqueeze(1)  # [B, 1, S, rot_dim]
    # batch_size, _, height, width, head_dim = freqs.shape
    batch_size, n_head, height, width, head_dim = q.shape
    # Apply rotary embeddings
    q_rotated = apply_rotary_emb(freqs, q).reshape(batch_size, n_head, height * width, head_dim)
    k_rotated = apply_rotary_emb(freqs, k).reshape(batch_size, n_head, height * width, head_dim)

    return q_rotated, k_rotated

def create_grids(batch_heights, batch_widths, n_dim, n_head):
    batch_size = len(batch_heights)
    max_height = max(batch_heights)
    max_width = max(batch_widths)
    grid_size = max_height * max_width

    # Create identical embeddings for all tokens
    embedding_vector = torch.randn(n_dim)  # Random vector for all tokens
    x_grid = embedding_vector.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, max_height, max_width, n_dim)

    # Reshape x_grid for multi-head attention [B, n_head, H, W, head_dim]
    x_grid = x_grid.reshape(batch_size, max_height, max_width, n_head, head_dim).permute(0, 3, 1, 2, 4)

    return x_grid

#%%

# Suppose we have batch data: input_ids of shape [B, S]
# Each item in the batch has a corresponding height and width
batch_heights = [5, 7, 6]
batch_widths =  [8, 4, 6]
max_seq_len = max(batch_heights) * max(batch_widths)

n_dim = 256  # Total embedding dimension
n_head = 8  # Number of attention heads
head_dim = n_dim // n_head
rot_dim = head_dim // 2  # Rotational dimension

grids = create_grids(batch_heights, batch_widths, n_dim, n_head)
q = grids
k = grids

pos_emb = RotaryEmbedding(
    dim=rot_dim,  # Set appropriately
    freqs_for='pixel',
    max_freq=max(max(batch_heights), max(batch_widths))
)

#%%
# pos_emb.number_of_axes = 2  # Manually set if not defined

# Get batched frequencies
batched_freqs = get_batched_freqs(pos_emb, batch_heights, batch_widths, max_seq_len, head_dim)  # Shape: [B, S, rot_dim]
q_rotated, k_rotated = apply_batched_rotary_emb(batched_freqs, q, k)
q_rotated.shape, k_rotated.shape
# %%

def position_to_index(row, col, width):
    return row * width + col

batch_idx = 0
width = batch_widths[batch_idx]
height = batch_heights[batch_idx]

# Pair 1 indices (Positions (0,0) and (1,1))
idx_A = position_to_index(0, 0, width)
idx_B = position_to_index(1, 2, width)

# Pair 2 indices (Positions (2,1) and (3,2))
idx_C = position_to_index(1, 3, width)
idx_D = position_to_index(2, 5, width)

q_A = q_rotated[batch_idx, :, idx_A, :].unsqueeze(0)
k_B = k_rotated[batch_idx, :, idx_B, :].unsqueeze(0)
q_C = q_rotated[batch_idx, :, idx_C, :].unsqueeze(0)
k_D = k_rotated[batch_idx, :, idx_D, :].unsqueeze(0)

# %%
q_A.shape, k_B.shape, q_C.shape, k_D.shape

# %%
# Compute attention scores for each pair
score_AB = torch.einsum('bhc,bhc->bh', q_A, k_B) / math.sqrt(head_dim)
score_CD = torch.einsum('bhc,bhc->bh', q_C, k_D) / math.sqrt(head_dim)

print("Attention scores between A and B:")
print(score_AB)

print("\nAttention scores between C and D:")
print(score_CD)

assert torch.allclose(score_AB, score_CD), "Attention scores between different pairs do not match."
# %%

## Test 2 (Vectorised version)

number_of_axes = 2
dim = head_dim // number_of_axes
rot_dim = dim * number_of_axes
max_height = max(batch_heights)
max_width = max(batch_widths)

# Initialize Rotary Embedding
pos_emb = RotaryEmbedding(
    dim=dim,
    freqs_for='pixel',
    max_freq=max(max_height, max_width)
)

max_freqs = pos_emb.get_axial_freqs(max_height, max_width)  # Shape: [max_height, max_width, rot_dim]
#%%
max_freqs_expanded = max_freqs.unsqueeze(0)  # Shape: [1, max_height, max_width, rot_dim]


# Generate batched indices
batch_size = len(batch_heights)
max_seq_len = max_height * max_width


grid_rows = torch.arange(max_height).unsqueeze(1).expand(max_height, max_width).flatten()  # [max_seq_len]
grid_cols = torch.arange(max_width).unsqueeze(0).expand(max_height, max_width).flatten()  # [max_seq_len]
grid_indices = torch.stack((grid_rows, grid_cols), dim=1)  # [max_seq_len, 2]

grid_indices.shape
# %%
batched_grid_indices = grid_indices.unsqueeze(0).expand(batch_size, -1, -1)  # [B, max_seq_len, 2]
batched_grid_indices.shape
# %%
sequence_lengths = torch.tensor([h * w for h, w in zip(batch_heights, batch_widths)])  # [B]
sequence_lengths
# %%
position_mask = torch.arange(max_seq_len).unsqueeze(0) < sequence_lengths.unsqueeze(1)  # [B, max_seq_len]
# position_mask
# %%
# Zero out invalid positions in indices (set to (0,0))
batched_grid_indices = batched_grid_indices * position_mask.unsqueeze(-1)
# %%
# batched_grid_indices
# %%
batched_freqs = max_freqs_expanded[0, batched_grid_indices[..., 0], batched_grid_indices[..., 1]]  # [B, max_seq_len, rot_dim]
batched_freqs.shape
# %%
batch_size = q.size(0)
q = q.reshape(batch_size, n_head, max_height*max_width, head_dim)
k = k.reshape(batch_size, n_head, max_height*max_width, head_dim)

q.shape, k.shape
# %%
# Apply rotary embeddings
q_rotated_1 = apply_rotary_emb(batched_freqs.unsqueeze(1), q)
k_rotated_1 = apply_rotary_emb(batched_freqs.unsqueeze(1), k)


q_A_ = q_rotated_1[batch_idx, :, idx_A, :].unsqueeze(0)
k_B_ = k_rotated_1[batch_idx, :, idx_B, :].unsqueeze(0)
q_C_ = q_rotated_1[batch_idx, :, idx_C, :].unsqueeze(0)
k_D_ = k_rotated_1[batch_idx, :, idx_D, :].unsqueeze(0)

score_AB_ = torch.einsum('bhc,bhc->bh', q_A_, k_B_) / math.sqrt(head_dim)
score_CD_ = torch.einsum('bhc,bhc->bh', q_C_, k_D_) / math.sqrt(head_dim)

print("Attention scores between A and B:")
print(score_AB_)

print("\nAttention scores between C and D:")
print(score_CD_)

assert torch.allclose(score_AB_, score_CD_), "Attention scores between different pairs do not match."
# %%

# %%
