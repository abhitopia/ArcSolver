from typing import Optional, Tuple
from torch import nn
from torch.nn import functional as F
import torch
from src.repl import REPLConfig, REPL
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List

def visualize_attention(attn_probs: List[List[torch.Tensor]], L: int, sampling_factor: int):
    """
    Visualizes attention probabilities from a model with multiple iterations and transformation blocks.

    Parameters:
    - attn_probs (List[List[torch.Tensor]]): Nested list where the outer list is iterations,
      the inner list is transformation blocks, and each Tensor is of shape [1, H, T, T].
      Assumes B=1.
    - L (int): Length of input tokens.
    - sampling_factor (int): Factor by which input and output tokens are sampled.

    Visualization Structure:
    - Rows: Iters * Blocks (each row corresponds to a specific transformation block within an iteration)
    - Columns: H (number of attention heads)
    - Each subplot: Heatmap of attention probabilities for a specific head
    - Token Labels: Correspond to original positions before sampling
    - Legend: Single color scale for the entire figure
    """
    
    # Validate input
    if not attn_probs:
        raise ValueError("attn_probs is empty.")
    
    Iters = len(attn_probs)
    Blocks = len(attn_probs[0])
    H = attn_probs[0][0].size(1)
    T = attn_probs[0][0].size(2)  # Assuming square attention matrices [T, T]
    
    # Function to sample tokens
    def sample_tokens(T: int, L: int, sampling_factor: int) -> List[int]:
        """
        Samples input and output tokens based on the sampling factor.
        Keeps all transformation tokens (0:3) intact.

        Returns:
        - List[int]: Indices of tokens to keep
        """
        # Transformation tokens: 0,1,2
        sampled_indices = list(range(0, 3))
        
        # Input tokens: 3 to L-1, sampled
        input_start = 3
        input_end = L
        sampled_input = list(range(input_start, input_end, sampling_factor))
        sampled_indices.extend(sampled_input)
        
        # Output tokens: L to T-1, sampled
        output_start = L
        output_end = T
        sampled_output = list(range(output_start, output_end, sampling_factor))
        sampled_indices.extend(sampled_output)
        
        return sampled_indices

    # Function to generate token labels based on sampled indices
    def generate_token_labels(sampled_indices: List[int], L: int) -> List[str]:
        """
        Generates token labels based on their original positions.

        Parameters:
        - sampled_indices (List[int]): List of sampled token indices
        - L (int): Length of input tokens

        Returns:
        - List[str]: List of token labels
        """
        labels = []
        for idx in sampled_indices:
            if idx < 3:
                labels.append(f'T{idx}')
            elif 3 <= idx < L:
                labels.append(f'I{idx}')
            else:
                labels.append(f'O{idx}')
        return labels

    # Sampled token indices
    sampled_indices = sample_tokens(T, L, sampling_factor)
    sampled_T = len(sampled_indices)  # New sequence length after sampling
    token_labels = generate_token_labels(sampled_indices, L)

    # Create subplot grid
    total_rows = Iters * Blocks
    total_cols = H

    # Initialize subplots
    fig = make_subplots(
        rows=total_rows,
        cols=total_cols,
        subplot_titles=[
            f'Iter {iter_idx+1} | Block {block_idx+1} | Head {head_idx+1}'
            for iter_idx in range(Iters)
            for block_idx in range(Blocks)
            for head_idx in range(H)
        ],
        horizontal_spacing=0.02,
        vertical_spacing=0.02
    )

    # Counter for subplot titles to manage color scale
    title_counter = 0

    # Iterate over iterations and blocks
    for iter_idx, iter_blocks in enumerate(attn_probs):
        for block_idx, block_attn in enumerate(iter_blocks):
            for head_idx in range(H):
                # Extract the attention matrix for the current head
                # Detach the tensor before converting to numpy to avoid RuntimeError
                attn_matrix = block_attn[0, head_idx, :, :].detach().cpu().numpy()  # Shape: [T, T]
                
                # Sample the attention matrix
                sampled_attn = attn_matrix[np.ix_(sampled_indices, sampled_indices)]  # Shape: [sampled_T, sampled_T]
                
                # Add Heatmap to the subplot
                fig.add_trace(
                    go.Heatmap(
                        z=sampled_attn,
                        x=token_labels,
                        y=token_labels,
                        colorscale='Viridis',
                        showscale=(title_counter == 0),  # Show color scale only for the first subplot
                        colorbar=dict(
                            title="Attention",
                            titleside="right",
                            titlefont=dict(size=10),
                            tickfont=dict(size=8)
                        ) if title_counter == 0 else None,
                        hovertemplate='Key: %{x}<br>Query: %{y}<br>Attention: %{z:.4f}<extra></extra>'
                    ),
                    row=iter_idx * Blocks + block_idx + 1,
                    col=head_idx + 1
                )
                
                title_counter += 1

    # Update layout
    fig.update_layout(
        height=300 * total_rows,  # Adjust height based on number of rows
        width=300 * total_cols,   # Adjust width based on number of columns
        title_text="Attention Probabilities Visualization",
        showlegend=False
    )

    # Update axes properties for all subplots
    for row in range(1, total_rows + 1):
        for col in range(1, total_cols + 1):
            fig.update_xaxes(
                title_text="Key Positions",
                tickangle=45,
                ticks='inside',
                tickfont=dict(size=8),
                showticklabels=True,
                row=row,
                col=col
            )
            fig.update_yaxes(
                title_text="Query Positions",
                ticks='inside',
                tickfont=dict(size=8),
                showticklabels=True,
                row=row,
                col=col
            )
    
    # Adjust subplot titles font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=8)

    # Show the figure
    fig.show()

    # We can save too using fig.write_html("attention.html")
    return fig




# Define the CustomSelfAttentionWithBias class
class CustomSelfAttentionWithBias(nn.Module):
    def __init__(self, config: REPLConfig, rope: Optional[None]=None, emphasis_factor: float=0.0, return_attn_scores: bool=False):
        super().__init__()
        self.config = config
        self.rope = rope
        self.return_attn_scores = return_attn_scores
        self.emphasis_factor = emphasis_factor  # Bias factor; set to 0.0 for no bias
        assert config.n_dim % config.n_head == 0, "n_dim must be divisible by n_head"
        
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.c_proj.RESCALE_RESIDUAL = 1

        # Regularization
        self.n_head = config.n_head
        self.n_dim = config.n_dim
        self.head_dim = config.n_dim // config.n_head
        self.dropout = config.dropout

        # Optional: Learnable bias
        self.learnable_bias = nn.Parameter(torch.zeros(1))  # Initialize to zero

    def forward(self,
                x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor]=None,
                positions: Optional[torch.Tensor]=None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]]=None, 
                return_kv_cache: bool=False,
                bias_transformation: bool=True) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        B, T, C = x.size()
        qkv = self.c_attn(x)  # qkv: (B, T, 3*C)
        q, k, v = qkv.split(self.n_dim, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Apply Rope2D to q and k (skipped as rope is None)
        if self.rope is not None and positions is not None:
            k = self.rope(k, positions.unsqueeze(1))
            q = self.rope(q, positions.unsqueeze(1))

        # If kv_cache is present, concatenate past keys and values
        if kv_cache is not None and isinstance(kv_cache, tuple):
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)  # (B, n_head, T_past + T, head_dim)
            v = torch.cat([past_v, v], dim=2)  # (B, n_head, T_past + T, head_dim)

        # Update new_kv_cache
        new_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = (k, v) if return_kv_cache else None

        # Compute scaled dot-product attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_head, T, T_k)

        # Apply bias to attention scores for transformation_id token (position 0)
        if bias_transformation:
            # Option A: Fixed Bias
            attn_scores[:, :, :, 0] += self.emphasis_factor

            # Option B: Learnable Bias
            # Uncomment the following line to use learnable bias instead
            # attn_scores[:, :, :, 0] += self.learnable_bias

        # Apply attention mask if provided
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # Assuming attn_mask shape is (B, 1, 1, T_k) or broadcastable to (B, n_head, T, T_k)
                attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
            else:
                # Assume attn_mask is additive and broadcastable
                attn_scores = attn_scores + attn_mask  # Assuming attn_mask is additive

        # Compute attention probabilities
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, n_head, T, T_k)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        # Compute attention output
        attn_output = torch.matmul(attn_probs, v)  # (B, n_head, T, head_dim)

        # Reshape back to (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(attn_output)

        if self.return_attn_scores:
            return y, attn_probs
        else:
            return y, new_kv_cache
        
def patch_model(model: REPL):

    sd = model.state_dict().copy()
    """
    Patch the model to use CustomSelfAttentionWithBias instead of the default attention.
    """
    for block in model.interpreter.blocks:
        attn = block.attn
        rope = attn.rope
        new_attn = CustomSelfAttentionWithBias(model.config, rope=rope, emphasis_factor=0.0, return_attn_scores=True)
        new_attn.load_state_dict(attn.state_dict(), strict=False)
        block.attn = new_attn
