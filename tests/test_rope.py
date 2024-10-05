#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory

#%%
import torch.nn as nn
from src.rope2d import RotaryPositionalEmbeddings, RoPE2D
from torch import Tensor
import torch

class SimpleSelfAttention(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 1024, is_2d: bool = False):
        super().__init__()
        self.head_dim = head_dim 

        # Linear layers for Q and K
        self.q_linear = nn.Linear(head_dim, head_dim, bias=False)
        self.k_linear = nn.Linear(head_dim, head_dim, bias=False)

        # Rotary Positional Embeddings
        if not is_2d:
            self.rope = RotaryPositionalEmbeddings(self.head_dim, max_seq_len=max_seq_len)
        else:
            self.rope = RoPE2D(dim=self.head_dim, max_height=max_seq_len, max_width=max_seq_len)

    def forward(self, x: Tensor, input_pos: Tensor) -> Tensor:
        """
        Computes attention scores using RoPE-enhanced Q and K.

        Args:
            x (Tensor): Input embeddings of shape [B, H, S, D].
            input_pos (Tensor): Position indices of shape [B, H, S].

        Returns:
            Tensor: Attention scores of shape [B, H, S, S].
        """
        # Compute Q and K
        q = self.q_linear(x)  # [B, H, S, D]
        k = self.k_linear(x)  # [B, H, S, D]

        # Apply RoPE
        q = self.rope(q, input_pos)  # [B, H, S, D]
        k = self.rope(k, input_pos)  # [B, H, S, D]

        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, S, S]
        return attn_scores
    


#%%

def test_rope_relative_distance():
    """
    Tests that attention scores between token pairs with the same relative positional distance are identical,
    regardless of their absolute positions.
    """
    # Set manual seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    head_dim = 8
    max_seq_len = 100

    # Initialize the attention module
    attention = SimpleSelfAttention(head_dim=head_dim, max_seq_len=max_seq_len, is_2d=False)

    # Define distinct token embeddings (one-hot vectors for clarity)
    token_A = torch.rand(head_dim)
    token_B = torch.rand(head_dim)
    token_C = token_A
    token_D = token_B

    num_heads = 1

    # Create two batches:
    # Batch 0: Tokens A and B at positions 
    # Batch 1: Tokens C and D at positions
    # Construct input embeddings tensor [B, H, S, D]
    embeddings_batch0 = torch.stack([token_A, token_B])  # [2, 4] (S, D)
    embeddings_batch1 = torch.stack([token_C, token_D])  # [2, 4] (S, D)
    embeddings = torch.stack([embeddings_batch0, embeddings_batch1])  # [2, 2, 4] (B, S, D)
    embeddings = embeddings.unsqueeze(1)  # [2, 1, 2, 4] (B, H, S, D)


    def test_a_pair(input_pair, should_match: bool):
        print("Input Position", input_pair)
        input_pos = torch.tensor(input_pair).unsqueeze(1)

        # Forward pass to get attention scores
        attn_scores = attention(embeddings, input_pos)  # [2, 1, 2, 2] (B, H, S, S)

        # Extract attention scores between the first and second tokens in each batch
        # Specifically, attn_scores[b, h, 0, 1] for each batch
        score_batch0 = attn_scores[0, 0, 0, 1].item()
        score_batch1 = attn_scores[1, 0, 0, 1].item()

        print(f"Attention score between Token A and B (Batch 0): {score_batch0:.6f}")
        print(f"Attention score between Token C and D (Batch 1): {score_batch1:.6f}")

        is_equal = torch.isclose(torch.tensor(score_batch0), torch.tensor(score_batch1), atol=1e-6)
        if should_match:
            assert is_equal, "Attention scores for equal relative distances should be identical."
        else:
            assert not is_equal, "Attention scores for non-equal relative distances should not be identical."
    

    # Test matching pair
    print("--Matching pair")
    matching_pair = [
        [4, 2],  # Batch 0 (A, B)
        [8, 6]   # Batch 1 (C, D)
    ]
    test_a_pair(input_pair=matching_pair, should_match=True)

    # Order matters
    print("--Order matters")
    order_swap_pair = [
        [2, 4],  # Batch 0 (A, B)
        [4, 2]   # Batch 1 (B, D)
    ]
    test_a_pair(input_pair=order_swap_pair, should_match=False)

    # Test non-matching pair
    print("--Non-matching pair")
    non_matching_pair = [
        [50, 3],  # Batch 0 (A, B)
        [3, 3]   # Batch 1 (C, D)
    ]
    test_a_pair(input_pair=non_matching_pair, should_match=False)


    # Visualise Attention Scores for a grid
    grid_width = 5
    grid_height = 5
    attention_score_ij = torch.zeros(grid_height, grid_width)
    for i in range(grid_height):
        for j in range(grid_width):
            embeddings_batch0 = torch.stack([token_A, token_B]).unsqueeze(0).unsqueeze(0)  # [1, 2, D] (1, 1, S, D)
            pos_ij = torch.tensor([[i, j]]).unsqueeze(0)  # [1, 1, 2] (1, 1, S)
            attn_scores = attention(embeddings_batch0, pos_ij)
            attention_score_ij[i, j] = attn_scores[0, 0, 0, 1].item()

    print(attention_score_ij)
    
    print("✅ Test Passed: Attention scores for identical relative distances are identical.")

test_rope_relative_distance()

#%%

#%%
# Test Case: Verify Attention Scores for Identical Relative 2D Distances
def test_rope2d_relative_distance():
    """
    Tests that attention scores between token pairs with the same relative 2D positional distances are identical,
    regardless of their absolute positions and distinct token embeddings.
    """
    # Set manual seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    head_dim = 8  # Must be even for RoPE2D
    num_heads = 1
    max_height = 100
    max_width = 100

    # Initialize the attention module
    attention = SimpleSelfAttention(
        head_dim=head_dim,
        max_seq_len=max(max_height, max_width),
        is_2d=True
    )

    # Define distinct token embeddings (one-hot vectors for clarity)
    # Each token embedding has size [head_dim * 2 = 8]
    token_A = torch.rand(head_dim)
    token_B = torch.rand(head_dim)
    token_C = token_A
    token_D = token_B


    # Create two batches:
    # Batch 0: Tokens A and B at positions (10, 20) and (12, 22) respectively (relative distance (2, 2))
    # Batch 1: Tokens C and D at positions (30, 40) and (32, 42) respectively (relative distance (2, 2))
    batch_size = 2
    num_heads = 2

    # Construct input embeddings tensor [B, H, S, D]
    # For simplicity, using the same token embeddings across heads
    embeddings_batch0 = torch.stack([token_A, token_B])  # [2, 8]
    embeddings_batch1 = torch.stack([token_C, token_D])  # [2, 8]
    embeddings = torch.stack([embeddings_batch0, embeddings_batch1])  # [2, 2, 8]
    embeddings = embeddings.unsqueeze(1).repeat(1, num_heads, 1, 1)  # [2, 2, 2, 8]

    # Define input_pos tensor [B, H, S, 2]
    # Batch 0: positions (10, 20) and (12, 22)
    # Batch 1: positions (30, 40) and (32, 42)
    positions = torch.tensor([
        [
            [[10, 20], [12, 22]],  # Head 0
            [[15, 20], [17, 25]]   # Head 1
        ],
        [
            [[30, 40], [32, 42]],  # Head 0
            [[30, 35], [32, 40]]   # Head 1
        ]
    ])  # [2, 2, 2, 2]

    # Forward pass to get attention scores
    attn_scores = attention(embeddings, positions)  # [2, 2, 2, 2]

    # Extract attention scores between the first and second tokens in each batch and each head
    # Specifically, attn_scores[b, h, 0, 1] for each batch and head
    score_batch0_head0 = attn_scores[0, 0, 0, 1].item()
    score_batch0_head1 = attn_scores[0, 1, 0, 1].item()
    score_batch1_head0 = attn_scores[1, 0, 0, 1].item()
    score_batch1_head1 = attn_scores[1, 1, 0, 1].item()

    print(f"Attention score between Token A and B (Batch 0, Head 0): {score_batch0_head0:.6f}")
    print(f"Attention score between Token A and B (Batch 0, Head 1): {score_batch0_head1:.6f}")
    print(f"Attention score between Token C and D (Batch 1, Head 0): {score_batch1_head0:.6f}")
    print(f"Attention score between Token C and D (Batch 1, Head 1): {score_batch1_head1:.6f}")

    # Assert that the attention scores are approximately equal for identical relative distances
    assert torch.isclose(torch.tensor(score_batch0_head0), torch.tensor(score_batch1_head0), atol=1e-6), \
        "Attention scores for equal relative distances across height should be identical across batches and heads."
    assert torch.isclose(torch.tensor(score_batch0_head1), torch.tensor(score_batch1_head1), atol=1e-6), \
        "Attention scores for equal relative distances across width should be identical across batches and heads."

    print("✅ Test Passed: Attention scores for identical relative 2D distances are identical across batches and heads.")

# # Execute the test
test_rope2d_relative_distance()

#%%

def test_neg_post_index():
      # Parameters
    head_dim = 8
    max_seq_len = 100

    # Define distinct token embeddings (one-hot vectors for clarity)
    token_A = torch.rand(head_dim).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 8]
    input_pos = torch.tensor([[-1]]).unsqueeze(0)  # [1, 1, 1, 2]
    rope = RotaryPositionalEmbeddings(head_dim, max_seq_len=max_seq_len)
    rotA = rope(token_A, input_pos)

    assert torch.allclose(token_A, rotA), "Rotary embeddings should not change for negative positional indices."
    print("✅ Test Passed: Rotary embeddings are unchanged for negative positional indices.")


    rope2d = RoPE2D(dim=head_dim, max_height=max_seq_len, max_width=max_seq_len)

    input_pos2d = torch.tensor([[-1, -1]]).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]

    rotA2d = rope2d(token_A, input_pos2d)

    assert torch.allclose(token_A, rotA2d), "Rotary embeddings should not change for negative positional indices in 2D RoPE."



test_neg_post_index()






# %%


# %%
