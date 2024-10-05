#%%
import torch
import torch.nn as nn
import numpy as np

class Rope2D(nn.Module):
    def __init__(self, h_dim, max_height, max_width, base=10000):
        super().__init__()
        self.head_dim = h_dim  # h_dim is the dimension per head
        self.max_height = max_height
        self.max_width = max_width
        self.base = base
        assert h_dim % 2 == 0, 'The head dimension must be divisible by 2.'
    
        # Exponential scaling
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer('inv_freq', inv_freq)
    
        # Precompute sinusoids
        h_sin, h_cos, w_sin, w_cos = self._build_sin_cos_tables()
        self.register_buffer('h_sin', h_sin)
        self.register_buffer('h_cos', h_cos)
        self.register_buffer('w_sin', w_sin)
        self.register_buffer('w_cos', w_cos)
    
    def _build_sin_cos_tables(self):
        # Position indices
        h_pos = torch.arange(self.max_height).unsqueeze(1)  # [max_height, 1]
        w_pos = torch.arange(self.max_width).unsqueeze(1)   # [max_width, 1]
    
        # Compute angles
        h_angles = h_pos * self.inv_freq.unsqueeze(0)  # [max_height, head_dim//2]
        w_angles = w_pos * self.inv_freq.unsqueeze(0)  # [max_width, head_dim//2]
    
        # Compute sin and cos values
        h_sin = torch.sin(h_angles)  # [max_height, head_dim//2]
        h_cos = torch.cos(h_angles)  # [max_height, head_dim//2]
        w_sin = torch.sin(w_angles)  # [max_width, head_dim//2]
        w_cos = torch.cos(w_angles)  # [max_width, head_dim//2]
    
        return h_sin, h_cos, w_sin, w_cos
    
    def forward(self, x, positions):
        """
        Args:
            x (Tensor): Input tensor of shape [batch_size, n_head, seq_length, head_dim].
            positions (Tensor): Positional indices of shape [batch_size, seq_length, 2],
                                where each position is (row, col).
        Returns:
            x_out (Tensor): Tensor with rotary embeddings applied, same shape as input x.
        """
        batch_size, n_head, seq_length, head_dim = x.shape
        assert head_dim == self.head_dim, "Head dimension mismatch."
    
        # Reshape x for rotary embeddings
        x = x.view(batch_size, n_head, seq_length, head_dim // 2, 2)
    
        # Extract positions
        h_pos = positions[..., 0]  # [batch_size, seq_length]
        w_pos = positions[..., 1]  # [batch_size, seq_length]
    
        # Gather sin and cos values based on positions
        h_sin = self.h_sin[h_pos]  # [batch_size, seq_length, head_dim // 2]
        h_cos = self.h_cos[h_pos]
        w_sin = self.w_sin[w_pos]
        w_cos = self.w_cos[w_pos]
    
        # Combine sin and cos values
        sin = torch.cat([h_sin, w_sin], dim=-1)  # [batch_size, seq_length, head_dim]
        cos = torch.cat([h_cos, w_cos], dim=-1)
    
        # Reshape sin and cos for broadcasting
        sin = sin.unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_length, head_dim, 1]
        cos = cos.unsqueeze(1).unsqueeze(-1)
    
        # x: [batch_size, n_head, seq_length, head_dim//2, 2]
        x1, x2 = x.unbind(-1)  # x1, x2: [batch_size, n_head, seq_length, head_dim // 2]
    
        # Apply rotary embeddings
        x_rotated = torch.stack([
            x1 * cos.squeeze(-1)[:, :, :, :head_dim // 2] - x2 * sin.squeeze(-1)[:, :, :, :head_dim // 2],
            x1 * sin.squeeze(-1)[:, :, :, :head_dim // 2] + x2 * cos.squeeze(-1)[:, :, :, :head_dim // 2]
        ], dim=-1)
    
        # Reshape back to original dimensions
        x_out = x_rotated.view(batch_size, n_head, seq_length, head_dim)
        return x_out



def test_rope2d_relative_position_invariance():
    # Parameters
    head_dim = 64
    n_head = 1  # Single attention head
    batch_size = 1
    seq_length = 1  # Testing individual positions
    max_height = 100
    max_width = 100
    base = 10000
    tolerance = 1e-5  # Tolerance for floating-point comparisons

    # Initialize Rope2D
    rope = Rope2D(h_dim=head_dim, max_height=max_height, max_width=max_width, base=base)

    # Create identical query vectors for positions A and C
    q_vector = torch.randn(batch_size, n_head, seq_length, head_dim)

    # Create identical key vectors for positions B and D
    k_vector = torch.randn(batch_size, n_head, seq_length, head_dim)

    # Positions where relative positions are the same
    # A = (x1, y1), B = (x2, y2)
    # C = (x3, y3), D = (x4, y4)
    # Ensure that (x1 - x2) == (x3 - x4) and (y1 - y2) == (y3 - y4)
    x1, y1 = 10, 20
    x2, y2 = 30, 51
    x3, y3 = 15, 25
    x4, y4 = 35, 55  # Adjusted y4 to be within valid range

    # Verify that positions are within bounds
    assert 0 <= x1 < max_height and 0 <= y1 < max_width, "Position A out of bounds"
    assert 0 <= x2 < max_height and 0 <= y2 < max_width, "Position B out of bounds"
    assert 0 <= x3 < max_height and 0 <= y3 < max_width, "Position C out of bounds"
    assert 0 <= x4 < max_height and 0 <= y4 < max_width, "Position D out of bounds"

    # Compute relative positions
    rel_pos_AB = (x1 - x2, y1 - y2)  # (-20, -30)
    rel_pos_CD = (x3 - x4, y3 - y4)  # (-20, -29)
    # Adjust x4 or y4 to ensure relative positions are the same
    y4 = y3 - (y1 - y2)  # y4 = 70 - (-30) = 100
    # Since y4 = 100 is out of bounds, adjust y4 to be within valid range
    y4 = y3 - (y1 - y2)  # y4 = 70 - (-30) = 100, adjust to 99
    y4 = min(max_width - 1, y4)

    # Recompute relative position
    rel_pos_CD = (x3 - x4, y3 - y4)
    # Confirm that relative positions are the same
    assert rel_pos_AB == rel_pos_CD, "Relative positions are not the same"

    # Prepare positions tensors
    # For positions A and B
    q_pos_AB = torch.tensor([[x1, y1]])
    k_pos_AB = torch.tensor([[x2, y2]])

    # For positions C and D
    q_pos_CD = torch.tensor([[x3, y3]])
    k_pos_CD = torch.tensor([[x4, y4]])

    # For positions with different relative positions
    x5, y5 = 10, 20
    x6, y6 = 35, 55  # (x5 - x6, y5 - y6) = (-25, -35)
    # Ensure positions are within bounds
    assert 0 <= x6 < max_height and 0 <= y6 < max_width, "Position with different relative position out of bounds"

    q_pos_diff = torch.tensor([[x5, y5]])
    k_pos_diff = torch.tensor([[x6, y6]])

    # Prepare position tensors for input
    q_positions_AB = q_pos_AB.unsqueeze(0)  # [batch_size, seq_length, 2]
    k_positions_AB = k_pos_AB.unsqueeze(0)

    q_positions_CD = q_pos_CD.unsqueeze(0)
    k_positions_CD = k_pos_CD.unsqueeze(0)

    q_positions_diff = q_pos_diff.unsqueeze(0)
    k_positions_diff = k_pos_diff.unsqueeze(0)

    # Apply positional encoding
    q_enc_AB = rope(q_vector, q_positions_AB)  # [batch_size, n_head, seq_length, head_dim]
    k_enc_AB = rope(k_vector, k_positions_AB)

    q_enc_CD = rope(q_vector, q_positions_CD)  # Using the same q_vector
    k_enc_CD = rope(k_vector, k_positions_CD)  # Using the same k_vector

    q_enc_diff = rope(q_vector, q_positions_diff)
    k_enc_diff = rope(k_vector, k_positions_diff)

    # Compute attention scores
    attn_score_AB = torch.einsum('bnhd,bnhd->bnh', q_enc_AB, k_enc_AB).item()
    attn_score_CD = torch.einsum('bnhd,bnhd->bnh', q_enc_CD, k_enc_CD).item()
    attn_score_diff = torch.einsum('bnhd,bnhd->bnh', q_enc_diff, k_enc_diff).item()

    print(f"Attention Score between A and B: {attn_score_AB}")
    print(f"Attention Score between C and D: {attn_score_CD}")
    print(f"Attention Score between positions with different relative positions: {attn_score_diff}")

    # Verify that attention scores between A and B and between C and D are the same
    difference_same = abs(attn_score_AB - attn_score_CD)
    print(f"Difference between attention scores (same relative positions): {difference_same}")
    if difference_same < tolerance:
        print("Test Passed: Attention scores are identical for same relative positions.")
    else:
        print("Test Failed: Attention scores differ for same relative positions.")

    # Verify that attention scores differ when relative positions are different
    difference_diff = abs(attn_score_AB - attn_score_diff)
    print(f"Difference between attention scores (different relative positions): {difference_diff}")
    if difference_diff > tolerance:
        print("Test Passed: Attention scores differ when relative positions differ.")
    else:
        print("Test Failed: Attention scores do not differ when relative positions differ.")

# Run the test
test_rope2d_relative_position_invariance()

# %%
