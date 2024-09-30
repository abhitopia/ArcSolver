#%%
import sys
from typing import List, Optional, Tuple
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory

import torch
from torch import nn, Tensor

from src.repl import REPLConfig, RMSNorm, TransformerBlock
#%%

# Sample configuration
config = REPLConfig(
    prog_vocab_size=1000,
    n_dim=64,
    n_embd=32,
    n_head=8,
    n_layer=2,
    n_state_layer=2,
    n_iter=4,  # Number of iterations
    dropout=0.0,
    pnorm=None
)

# Generate sample input
B = 2  # Batch size
T = 5  # Sequence length within each tensor
D = config.n_dim  # Feature dimension

torch.manual_seed(42)
# Create a list of tensors representing states at each iteration
iter_states = []
for i in range(config.n_iter):
    # Each state is a tensor of shape (B, T, D)
    state = torch.randn(B, T, D)
    iter_states.append(state)




#%%

class StateAggregator(nn.Module):
    def __init__(self, config: REPLConfig) -> None:
        super().__init__()
        self.config = config
        self.n_state_layer = config.n_state_layer
        self.pos_emb = nn.Embedding(config.n_iter+1, config.n_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config, rope=None) for _ in range(self.n_state_layer)])
        self.rms_out = RMSNorm(config.n_dim)

    def get_causal_mask(self, qT: int, kT: int, device: torch.device):
        offset = kT - qT
        causal_mask = torch.ones(1, qT, kT, dtype=torch.bool, device=device).tril(diagonal=offset).unsqueeze(1)
        return causal_mask

    def forward_nocache(self, x: List[Tensor]) -> Tensor:
        B, T, D = x[0].size()
        x_cat = torch.stack(x, dim=2) # (B, T, n_iter, D)
        x_cat = x_cat.reshape(B*T, -1, D)
        n_iters = x_cat.size(1)

        x_pos = torch.arange(n_iters, device=x[0].device).unsqueeze(0)
        x_pos_emb = self.pos_emb(x_pos)

        x_cat = x_cat + x_pos_emb
        attn_mask = self.get_causal_mask(n_iters, n_iters, x_cat.device)

        for idx, block in enumerate(self.blocks):
            x_cat, _ = block(x_cat,
                        attn_mask=attn_mask,
                        positions=None,
                        return_kv_cache=False)
        
        # Reshape back to (B, T, C)
        output = x_cat[:, -1, :].reshape(B, T, D)
        output = self.rms_out(output)
        return output, None
    
    def forward(self, x: List[Tensor], kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None):
        B, T, D = x[-1].size()
        past_iters = 0 if kv_caches is None else kv_caches[0][0].size(2)
        n_iters = len(x)
        new_iters = n_iters - past_iters
        x_cat = torch.stack(x[-new_iters:], dim=2) # (B, T, new_iters, D)
        x_cat = x_cat.reshape(B*T, -1, D)
        x_pos = torch.arange(past_iters, n_iters, device=x[0].device).unsqueeze(0)
        x_pos_emb = self.pos_emb(x_pos)

        x_cat = x_cat + x_pos_emb

        attn_mask = self.get_causal_mask(new_iters, n_iters, x_cat.device)
        # NOTE: In incremental setting, even if attn_mask is None, (full non-causal attention)
        # The fact that previous states are cached and don't access the future states means that
        # the model is still causal.
        # attn_mask: Optional[torch.Tensor] = None


        updated_kv_caches: List[Tuple[Tensor, Tensor]] = []
        for idx, block in enumerate(self.blocks):
            x_cat, kv_cache = block(
                                x_cat,
                                attn_mask=attn_mask,
                                positions=None,
                                kv_cache=kv_caches[idx] if kv_caches is not None else None,
                                return_kv_cache=True)
            
            if kv_cache is not None:
                updated_kv_caches.append(kv_cache)

        # Reshape back to (B, T, C)
        output = x_cat[:, -1, :].reshape(B, T, D)
        output = self.rms_out(output)

        return output, updated_kv_caches



# Initialize the StateAggregator
state_agg = StateAggregator(config)
state_agg.eval()  # Set to evaluation mode to disable dropout


# Run the forward pass
output_nocache, _ = state_agg.forward_nocache(iter_states)

output_all, _ = state_agg(iter_states)

cache = None
for i in range(1, len(iter_states)+1):
    output, cache = state_agg(iter_states[:i], cache)


# print("Output shape:", output.shape)
# print("Output tensor:", output.sum())
print("Output tensor:", output_nocache.sum())
print("Output tensor:", output_all.sum())

# output, _ = state_agg(iter_states)

# # Print the output
# print("Output shape:", output.shape)
# print("Output tensor:", output.sum())
# print("Output tensor:", output_nocache.sum())
# %%
