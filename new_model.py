#%%
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from src.rope_2d import Rope2D
from src.interpreter import RMSNorm, SwiGLUFFN
from src.tokenizer import MODEL_INPUT, MODEL_OUTPUT, ArrayTransformTokenizer, ColorPermutationTokenizer, GridTokenizer
from src.mask_utils import create_enc_dec_mask

def create_test_inp(bs=10, inp_seq_len=10, out_seq_len=5, prog_vocab_size=15, perm_vocab_size=10, tform_vocab_size=11, grid_vocab_size=16, pad_idx=0):

    program = torch.randint(0, prog_vocab_size, (bs, 1))
    color_permutation = torch.randint(0, perm_vocab_size, (bs, 1))
    array_transform = torch.randint(0, tform_vocab_size, (bs, 1))
    input_grid = torch.randint(1, grid_vocab_size, (bs, inp_seq_len))
    output_grid = torch.randint(1, grid_vocab_size, (bs, out_seq_len))

    def random_indices(length):
        indices = np.random.randint(0, grid_vocab_size, (length, 2))
        return [(i, j) for i, j in indices]

    inp_indices = torch.full((bs, inp_seq_len, 2), -1, dtype=torch.int64)
    out_indices = torch.full((bs, out_seq_len, 2), -1, dtype=torch.int64)
    for b in range(bs):
        inp_len = np.random.randint(inp_seq_len//2, inp_seq_len)
        input_grid[b, inp_len:] = pad_idx
        inp_indices[b, :inp_len, :] = torch.tensor(random_indices(inp_len), dtype=torch.long)
        out_len = np.random.randint(out_seq_len//2, out_seq_len)
        output_grid[b, out_len:] = pad_idx
        out_indices[b, :out_len, :] = torch.tensor(random_indices(out_len), dtype=torch.long)


    x = MODEL_INPUT(program=program,
                color_permutation=color_permutation,
                array_transform=array_transform, 
                grid=input_grid, 
                grid_indices=inp_indices,
                meta=None)
    y = MODEL_OUTPUT(grid=output_grid,
                grid_indices=out_indices)
    return x, y


# %%
@dataclass
class REPLConfig:
    prog_vocab_size: int # number of program tokens
    n_dim: int  # dimension of the model
    n_embd: int # embedding dimension
    n_head: int # number of heads within each self-attention block
    n_layer: int = 1 # number of transformer blocks / layers
    pnorm: float = None
    dropout: float = 0.0 # dropout probability
    grid_vocab_size: int = len(GridTokenizer()) # number of array element tokens (one extra for niceness)
    perm_vocab_size: int = len(ColorPermutationTokenizer())
    tform_vocab_size: int = len(ArrayTransformTokenizer())
    max_iters: int = 64 # maximum number of iterations
    n_state_layer: int = 1 # number of transformer blocks / layers for state aggregation    

    def __post_init__(self):
        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        if self.pnorm is not None:
            assert 0.0 < self.pnorm, "p-norm must be greater than 0"
        
        head_dim = self.n_dim // self.n_head
        assert head_dim % 2 == 0, "Head dimension must be even"


    def to_dict(self):
        return {
            'prog_vocab_size': self.prog_vocab_size,
            'n_dim': self.n_dim,
            'n_embd': self.n_embd,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'pnorm': self.pnorm,
            'dropout': self.dropout,
            'grid_vocab_size': self.grid_vocab_size,
            'perm_vocab_size': self.perm_vocab_size,
            'tform_vocab_size': self.tform_vocab_size,
            'max_iters': self.max_iters,
            'n_state_layer': self.n_state_layer
        }
    
    @staticmethod
    def from_dict(data: dict) -> "REPLConfig":
        return REPLConfig(**data)

# %%

from torch import Tensor
from torch.nn import functional as F

class SelfAttention(nn.Module):
    def __init__(self, config: REPLConfig, rope: Optional[Rope2D]=None):
        super().__init__()
        self.config = config
        self.rope = rope
        assert config.n_dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.c_proj.RESCALE_RESIDUAL = 1

        # regularization
        self.n_head = config.n_head
        self.n_dim = config.n_dim
        self.dropout = config.dropout


    # def forward(self, x, attn_mask=None, kv_cache=None, return_kv_cache=False):

    def forward(self,
            x: Tensor, 
            attn_mask: Tensor,
            positions: Optional[Tensor] = None,
            kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        
        B, T, C = x.size() 
        qkv = self.c_attn(x)        # qkv: (B, T, 3 * C)
        q, k, v = qkv.split(self.n_dim, dim=2)

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T,  head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  

        # Apply Rope2D to q and k
        if self.rope is not None:
            k = self.rope(k, positions)
            q = self.rope(q, positions)


        # # If kv_cache is present, concatenate past keys and values
        # Assume kv_cache is has rope applied to it already
        if kv_cache is not None and torch.jit.isinstance(kv_cache, Tuple[Tensor, Tensor]):
            past_k, past_v = kv_cache  # K: (B, n_head, T_past, head_dim)
            k = torch.cat([past_k, k], dim=2)  # Concatenate along sequence length dimension
            v = torch.cat([past_v, v], dim=2)
            # print("Past K", past_k.size(), "New K", k.size())

        # Update new_kv_cache
        new_kv_cache: Optional[Tuple[Tensor, Tensor]] = (k, v) if return_kv_cache else None

        # Compute attention
        dropout_p = self.dropout if self.training else 0.0

        # attn_output: (B, n_head, T, head_dim)
        # print("QKV", q.size(), k.size(), v.size(), attn_mask.size())
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        # Reshape back to (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(attn_output)

        # Zero out NaN values, so they don't affect future computations
        # I have also verified that the it doesn't matter what the nan values are set to
        y = torch.nan_to_num(y, nan=0.0)

        return y, new_kv_cache

class TransformerBlock(nn.Module):
    def __init__(self, config: REPLConfig, rope: Optional[Rope2D]):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.rmsnorm = RMSNorm(config.n_dim)
        self.attn = SelfAttention(config, rope)
        self.normed_mlp = nn.Sequential(
                            RMSNorm(config.n_dim),
                            SwiGLUFFN(config.n_dim, 4 * config.n_dim))

    def forward(self, x: Tensor, 
            attn_mask: Tensor, 
            positions: Optional[Tensor] = None,
            kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:

        attn_output, new_kv_cache = self.attn(self.rmsnorm(x), attn_mask=attn_mask, positions=positions, kv_cache=kv_cache, return_kv_cache=return_kv_cache)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.normed_mlp(x))
        return x, new_kv_cache

class StateAggregator(nn.Module):
    def __init__(self, config: REPLConfig) -> None:
        super().__init__()
        self.config = config
        self.n_state_layer = config.n_state_layer
        self.pos_emb = nn.Embedding(config.max_iters, config.n_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config, rope=None) for _ in range(self.n_state_layer)])
        self.rms_out = RMSNorm(config.n_dim)

    def get_causal_mask(self, qT: int, kT: int, device: torch.device):
        offset = kT - qT
        causal_mask = torch.ones(1, qT, kT, dtype=torch.bool, device=device).tril(diagonal=offset).unsqueeze(1)
        return causal_mask

    def forward_nocache(self, x: List[Tensor]) -> Tensor:
        B, T, D = x[0].size()
        x_reshape = [i.reshape(B*T, D) for i in x]
        x_cat = torch.stack(x_reshape, dim=0).permute(1, 0, 2)
        n_iters = x_cat.size(1)
        x_pos = torch.arange(n_iters, device=x[0].device).unsqueeze(0)
        x_pos_emb = self.pos_emb(x_pos)
        x_cat = x_cat + x_pos_emb
        attn_mask = self.get_causal_mask(n_iters, n_iters, x_cat.device)
        # attn_mask = None
        print("Mask", attn_mask)
        # print("Mask", attn_mask)
        for block in self.blocks:
            x_cat, _ = block(x_cat,
                        attn_mask=attn_mask,
                        positions=None,
                        return_kv_cache=False)

        # Reshape back to (B, T, C)
        output = x_cat[:, -1, :].reshape(B, T, D)
        output = self.rms_out(output)
        return output
    

    def forward(self, x: List[torch.Tensor], kv_caches: Optional[List[Tuple[Tensor, Tensor]]]=None):
        B, T, D = x[-1].size()
        past_iters = 0 if kv_caches is None else kv_caches[0][0].size(2)
        n_iters = len(x)
        new_iters = n_iters - past_iters
        x_reshape = [i.reshape(B*T, D) for i in x[-new_iters:]]
        x_cat = torch.stack(x_reshape, dim=0).permute(1, 0, 2)
        x_pos = torch.arange(past_iters, n_iters, device=x[0].device).unsqueeze(0)
        x_pos_emb = self.pos_emb(x_pos)
        x_cat = x_cat + x_pos_emb

        # attn_mask = self.get_causal_mask(new_iters, n_iters, x_cat.device)
        # NOTE: In incremental setting, even if attn_mask is None, (full non-causal attention)
        # The fact that previous states are cached and don't access the future states means that
        # the model is still causal.
        attn_mask = None

        if kv_caches is None:
            kv_caches = [None] * self.n_state_layer

        updated_kv_caches: List[Tuple[Tensor, Tensor]] = []
        for idx, block in enumerate(self.blocks):
            # print("Block", idx, x_cat.size())
            x_cat, kv_cache = block(
                                x_cat,
                                attn_mask=attn_mask,
                                positions=None,
                                kv_cache=kv_caches[idx],
                                return_kv_cache=True)
            # print("Mask", attn_mask.shape)
            updated_kv_caches.append(kv_cache)

        # Reshape back to (B, T, C)
        output = x_cat[:, -1, :].reshape(B, T, D)
        output = self.rms_out(output)
        return output, updated_kv_caches

class Interpreter(nn.Module):
    def __init__(self, config: REPLConfig) -> None:
        super().__init__()
        self.config = config

        rope_2d = Rope2D(config.n_dim // config.n_head, max_height=60, max_width=60)
        self.blocks = nn.ModuleList([TransformerBlock(config, rope=rope_2d) for _ in range(config.n_layer)])
        
    def forward(self,
            x: Tensor, 
            attn_mask: Tensor, 
            positions: Tensor, 
            kv_cache: Optional[List[Tuple[Tensor, Tensor]]] = None,
            return_kv_caches: bool = False
        ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:

        loop_kv_caches: List[Tuple[Tensor, Tensor]] = []

        for i, block in enumerate(self.blocks):
            x, new_kv_cache = block( x, 
                attn_mask=attn_mask,
                positions=positions, 
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                return_kv_cache=return_kv_caches
            )

            # Ensure new_kv_cache is not None before appending
            if return_kv_caches and new_kv_cache is not None:
                loop_kv_caches.append(new_kv_cache)

        return x, loop_kv_caches

class REPL(nn.Module):
    def __init__(self,
                config: REPLConfig,
                pad_idx: int = 0,
            ):
        super().__init__()
        self.config = config
        self.n_dim = config.n_dim
        self.n_layer = config.n_layer
        self.pnorm = config.pnorm

        self.PAD_IDX = pad_idx

        self.pte = nn.Sequential(
            nn.Embedding(config.prog_vocab_size, config.n_embd, sparse=True),
            nn.Linear(config.n_embd, config.n_dim, bias=False)
        )

        self.cte = nn.Sequential(
            nn.Embedding(config.perm_vocab_size, config.n_embd),
            nn.Linear(config.n_embd, config.n_dim, bias=False)
        )

        self.ate = nn.Sequential(
            nn.Embedding(config.tform_vocab_size, config.n_embd),
            nn.Linear(config.n_embd, config.n_dim, bias=False)
        )

        self.gte = nn.Sequential(
            nn.Embedding(config.grid_vocab_size, config.n_embd),
            nn.Linear(config.n_embd, config.n_dim, bias=False)
        )

        # color + array + program + inp_grid + pad + out_grid
        self.type_emb = nn.Embedding(5, config.n_dim)

        dummy_idx = torch.ones((1, 1), dtype=torch.long, requires_grad=False)
        self.register_buffer('prog_type_idx', (0 * dummy_idx.clone()))
        self.register_buffer('color_type_idx', (1 * dummy_idx.clone()))
        self.register_buffer('tform_type_idx', (2 * dummy_idx.clone()))
        self.register_buffer('inp_grid_type_idx', (3 * dummy_idx.clone()))
        self.register_buffer('out_grid_type_idx', (4 * dummy_idx.clone()))


        # rope_2d = Rope2D(config.n_dim // config.n_head, max_height=60, max_width=60)
        # self.blocks = nn.ModuleList([TransformerBlock(config, rope=rope_2d) for _ in range(config.n_layer)])
        
        # self.inp_inject = nn.Linear(2*config.n_dim, config.n_dim, bias=False)

        # self.ln_f = RMSNorm(config.n_dim)
        self.interpreter = Interpreter(config)
        self.state_agg = StateAggregator(config)

        self.lm_head = nn.Linear(config.n_dim, config.grid_vocab_size, bias=False)
                
        # weight sharing scheme. Transformer++ (Llama architecture does not tie weights)
        # Reference: https://youtu.be/pRM_P6UfdIc?t=1500
        # self.wte.weight = self.lm_head.weight

        # init params
        # self.apply(self._init_weights)
        self.init_prog_embedding()

    def init_prog_embedding(self):
        if self.pnorm is not None:
            """Initialize prog embedding vectors with the target L2 norm."""
            with torch.no_grad():
                # Initialize the embedding weights to random values
                nn.init.normal_(self.pte[0].weight)
                # Normalize each embedding vector to the target L2 norm
                norm = self.pte[0].weight.norm(p=2, dim=1, keepdim=True)
                self.pte[0].weight = nn.Parameter(self.pte[0].weight * (self.pnorm / norm))

    def contruct_encoder_input(self, x: MODEL_INPUT):
        program = self.pte(x.program)
        program_type = self.type_emb(self.prog_type_idx)
        program = program + program_type

        color_permutation = self.cte(x.color_permutation)
        color_type = self.type_emb(self.color_type_idx)
        color_permutation = color_permutation + color_type

        array_transform = self.ate(x.array_transform)
        array_type = self.type_emb(self.tform_type_idx)
        array_transform = array_transform + array_type

        inp_grid = self.gte(x.grid)
        inp_grid_type = self.type_emb(self.inp_grid_type_idx)
        inp_grid = inp_grid + inp_grid_type

        grid_valid_mask = x.grid != self.PAD_IDX
        grid_indices = x.grid_indices

        pca_valid_mask = torch.ones((x.grid.size(0), 3), dtype=torch.bool, device=x.grid.device)
        pca_indices = torch.full((x.grid.size(0), 3, 2), -1, dtype=grid_indices.dtype)

        enc_valid_mask = torch.cat([pca_valid_mask, grid_valid_mask], dim=1)
        enc_indices = torch.cat([pca_indices, grid_indices], dim=1)

        enc_inp = torch.cat([program, color_permutation, array_transform, inp_grid], dim=1)
        return enc_inp, enc_valid_mask, enc_indices

    def contruct_decoder_input(self, y: MODEL_OUTPUT):
        out_grid = self.gte(y.grid)
        out_grid_type = self.type_emb(self.out_grid_type_idx)
        dec_inp = out_grid + out_grid_type
        dec_valid_mask = y.grid != self.PAD_IDX
        dec_indices = y.grid_indices
        return dec_inp, dec_valid_mask, dec_indices
    
    def forward(self, x: MODEL_INPUT,
            y: Optional[MODEL_OUTPUT] = None, 
            iters: int = 1, 
            return_cache: bool = False
        )-> Tuple[List[Tensor], Optional[Tuple[List[List[Tuple[Tensor, Tensor]]], Tensor, Tensor]]]:

        if y is None:
            bs = x.grid.size(0)
            y = MODEL_OUTPUT(
                    grid=torch.zeros((bs, 0), dtype=x.grid.dtype, device=x.grid.device),
                    grid_indices=torch.zeros((bs, 0, 2), dtype=x.grid_indices.dtype, device=x.grid.device))
                                                 
        enc_inp, enc_valid_mask, enc_indices = self.contruct_encoder_input(x)
        dec_inp, dec_valid_mask, dec_indices = self.contruct_decoder_input(y)

        attn_mask = create_enc_dec_mask(enc_valid_mask, dec_valid_mask).unsqueeze(1)

        dec_start_idx = enc_inp.size(1)

        enc_dec_inp = torch.cat([enc_inp, dec_inp], dim=1)
        enc_dec_indices = torch.cat([enc_indices, dec_indices], dim=1)

        logits = []
        iter_states = [enc_dec_inp]

        updated_kv_cache: Optional[List[List[Tuple[Tensor, Tensor]]]] = [] if return_cache else None

        current_state, states_kv_cache = self.state_agg(iter_states, None)
        for i in range(iters):
            new_state, iter_kv_cache = self.interpreter(
                                            x=current_state,
                                            attn_mask=attn_mask,
                                            positions=enc_dec_indices,
                                            kv_cache=None,
                                            return_kv_caches=return_cache)
            iter_states.append(new_state)
            current_state, states_kv_cache = self.state_agg(iter_states, states_kv_cache)
            logits_i = self.lm_head(current_state[:, dec_start_idx:, :])
            logits.append(logits_i)

            if updated_kv_cache is not None:
                updated_kv_cache.append(iter_kv_cache)

        cache = (updated_kv_cache, enc_valid_mask, dec_valid_mask) if return_cache else None
        return logits, cache

    def forward_inc(self,
            next_y: MODEL_OUTPUT, 
            cache: Tuple[List[List[Tuple[Tensor, Tensor]]], Tensor, Tensor],
            iters: int = 1, 
        ) -> Tuple[List[Tensor], Optional[List[List[Tuple[Tensor, Tensor]]]]]:
                   
        dec_inp, dec_valid_mask, dec_indices = self.contruct_decoder_input(next_y)
        seq_len = dec_inp.size(1)
        past_kv_cache, past_enc_valid_mask, past_dec_valid_mask = cache

        dec_valid_mask = torch.cat([past_dec_valid_mask, dec_valid_mask], dim=1)

        attn_mask = create_enc_dec_mask(past_enc_valid_mask, dec_valid_mask).unsqueeze(1)

        # Need to strip the attn_mask to match the size of decoder input
        attn_mask = attn_mask[:, :, -seq_len:, :]

        logits = []
        iter_states = [dec_inp]

        updated_kv_caches: List[List[Tuple[Tensor, Tensor]]] = []
        current_state, states_kv_cache = self.state_agg(iter_states, None)

        for i in range(iters):
            new_state, iter_kv_cache = self.interpreter(
                                            x=current_state,
                                            attn_mask=attn_mask,
                                            positions=dec_indices,
                                            kv_cache=past_kv_cache[i],
                                            return_kv_caches=True)

            iter_states.append(new_state)
            current_state, states_kv_cache = self.state_agg(iter_states, states_kv_cache)
            logits_i = self.lm_head(current_state)
            logits.append(logits_i)

            # Store the updated kv-cache for this loop iteration
            updated_kv_caches.append(iter_kv_cache)

        return logits, updated_kv_caches



# logitsx = model.forwardx(x, y, iters=3)
# %%

config = REPLConfig(
        prog_vocab_size=15,
        n_dim=128,
        n_embd=16, 
        n_head=8,
        n_layer=3, 
        pnorm=2.0, 
        dropout=0.0,
        n_state_layer=3,
    )


x, y = create_test_inp(
    bs=3,
    inp_seq_len=4, 
    out_seq_len=4,
    grid_vocab_size=config.grid_vocab_size,
    tform_vocab_size=config.tform_vocab_size,
    perm_vocab_size=config.perm_vocab_size,
    prog_vocab_size=config.prog_vocab_size)

x.grid, x.grid_indices, y.grid, y.grid_indices

valid_mask = torch.cat([(x.grid != 0), (y.grid != 0)], dim=1)
print("Valid Mask", valid_mask)
model = REPL(config)

#%%
logits, cache = model(x, None, iters=4, return_cache=True)
logits[-1][:, :, 0], y.grid.shape

#%%
#%%
logits_next, cache_next = model.forward_inc(y, cache, iters=4)
logits_next[-1][:, :, 0], y.grid.shape
#%%
logits, cache = model(x, y, iters=4, return_cache=True)
logits[-1][:, :, 0], y.grid.shape
#%%
global_nan_value = 0.0
logits, cache = model(x, y, iters=4, return_cache=True)# %%
logits[-1][:, :, 0]
# logitsx[0]
# %%
global_nan_value = 0.0
logits = model.forwardb(x, y, iters=4)
logits[-1][:, :, 0]
#
#%%
len(cache[0][0]), cache[0][0][1].size()
# %%
# %%


# %%
