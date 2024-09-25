#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory


import torch
import math
from src.rope import RotaryEmbedding, apply_rotary_emb

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

    for i in range(batch_size):
        height = batch_heights[i]
        width = batch_widths[i]
        freqs = get_item_freqs(pos_emb, height, width, head_dim)
        freqs_padded = pad_freqs(freqs, max_seq_len) # Shape: [max_seq_len, head_dim]
        batched_freqs[i] = freqs_padded

    return batched_freqs  # Shape: [batch_size, max_seq_len, head_dim]


# Assume q and k are of shape [B, N_HEADS, S, HEAD_DIM]
def apply_batched_rotary_emb(freqs, q, k, start_index=0):
    batch_size, n_heads, seq_len, head_dim = q.shape
    rot_dim = freqs.shape[-1]

    # Ensure rot_dim does not exceed head_dim - start_index
    assert start_index + rot_dim <= head_dim, "Rotational dimensions exceed head dimensions."

    # Reshape freqs to match [B, 1, S, rot_dim]
    freqs = freqs.unsqueeze(1)  # [B, 1, S, rot_dim]

    # Apply rotary embeddings
    q_rotated = apply_rotary_emb(freqs, q, start_index=start_index)
    k_rotated = apply_rotary_emb(freqs, k, start_index=start_index)

    return q_rotated, k_rotated

#%%

# Suppose we have batch data: input_ids of shape [B, S]
# Each item in the batch has a corresponding height and width
batch_heights = [3, 5, 2, 4]
batch_widths =  [5, 1, 2, 3]
max_seq_len = max(batch_heights) * max(batch_widths)

n_dim = 256  # Total embedding dimension
n_head = 4  # Number of attention heads
head_dim = n_dim // n_head
rot_dim = head_dim // 2  # Rotational dimension


pos_emb = RotaryEmbedding(
    dim=rot_dim,  # Set appropriately
    freqs_for='pixel',
    max_freq=max(max(batch_heights), max(batch_widths))
)

# pos_emb.number_of_axes = 2  # Manually set if not defined

# Get batched frequencies
batched_freqs = get_batched_freqs(pos_emb, batch_heights, batch_widths, max_seq_len, head_dim)  # Shape: [B, S, rot_dim]

# %%


q_rotated, k_rotated = apply_batched_rotary_emb(batched_freqs, q, k, start_index=0)

# %%
