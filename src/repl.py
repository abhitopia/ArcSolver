#%%
from dataclasses import dataclass
import re
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .lazy_adamw import LazyAdamW
from .rope_2d import Rope2D
from .tokenizer import MODEL_INPUT, MODEL_OUTPUT, ArrayTransformTokenizer, ColorPermutationTokenizer, GridTokenizer
from .mask_utils import create_enc_dec_mask
from .multilevel_loss import MultiLevelLoss
from .utils import get_logger

logger = get_logger()

# %%
@dataclass
class REPLConfig:
    prog_vocab_size: int # number of program tokens
    n_dim: int  # dimension of the model
    n_embd: int # embedding dimension
    n_head: int # number of heads within each self-attention block
    n_layer: int = 1 # number of transformer blocks / layers
    pnorm: Optional[float] = None
    dropout: float = 0.0 # dropout probability
    grid_vocab_size: int = len(GridTokenizer()) # number of array element tokens (one extra for niceness)
    perm_vocab_size: int = len(ColorPermutationTokenizer())
    tform_vocab_size: int = len(ArrayTransformTokenizer())
    n_iter: int = 8 # number of iterations
    n_state_layer: int = 1 # number of transformer blocks / layers for state aggregation    
    pad_idx: int = GridTokenizer().PAD_IDX
    max_grid_height: int = 60
    max_grid_width: int = 60

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
            'n_iter': self.n_iter,
            'n_state_layer': self.n_state_layer,
            'pad_idx': self.pad_idx,
            'max_grid_height': self.max_grid_height,
            'max_grid_width': self.max_grid_width
        }
    
    @staticmethod
    def from_dict(data: dict) -> "REPLConfig":
        return REPLConfig(**data)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class SwiGLUFFN(nn.Module):
    """SwiGLUFFN

    Taken from: https://github.com/kyegomez/zeta/tree/master
    Args:
        nn (_type_): _description_

    Examples:
    >>> import torch
    >>> x = torch.randn(5, 10)
    >>> swiglu = SwiGLUFFN(10, 20)
    >>> swiglu(x).shape
    torch.Size([5, 10])
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

        # Note that it adds extra params, but I don't care about it.
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return x


class RMSNorm(nn.Module):
    """
    Ref Source: https://pytorch.org/torchtune/stable/_modules/torchtune/modules/rms_norm.html#RMSNorm
    Implements Root Mean Square Normalization introduced in
    https://arxiv.org/abs/1910.07467.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor to normalize

        Returns:
            Tensor: The output tensor after applying RMSNorm.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (
            x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        ).type_as(x)
        return x_normed * self.scale



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
            attn_mask: Optional[Tensor],
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
        if self.rope is not None and positions is not None:
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
        # y = torch.nan_to_num(y, nan=0.0)

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
            attn_mask: Optional[Tensor], 
            positions: Optional[Tensor] = None,
            kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:

        attn_output, new_kv_cache = self.attn(self.rmsnorm(x), attn_mask=attn_mask, positions=positions, kv_cache=kv_cache, return_kv_cache=return_kv_cache)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.normed_mlp(x))
        return x, new_kv_cache


# class StateAggregator(nn.Module):
#     def __init__(self, config: REPLConfig) -> None:
#         super().__init__()
#         self.config = config
#         self.n_state_layer = config.n_state_layer
#         self.pos_emb = nn.Embedding(config.n_iter+1, config.n_dim)
#         self.blocks = nn.ModuleList([TransformerBlock(config, rope=None) for _ in range(self.n_state_layer)])
#         self.rms_out = RMSNorm(config.n_dim)

#     def get_causal_mask(self, qT: int, kT: int, device: torch.device):
#         offset = kT - qT
#         causal_mask = torch.ones(1, qT, kT, dtype=torch.bool, device=device).tril(diagonal=offset).unsqueeze(1)
#         return causal_mask

#     def forward_nocache(self, x: List[Tensor]) -> Tensor:
#         B, T, D = x[0].size()
#         x_cat = torch.stack(x, dim=2) # (B, T, n_iter, D)
#         x_cat = x_cat.reshape(B*T, -1, D)
#         n_iters = x_cat.size(1)

#         x_pos = torch.arange(n_iters, device=x[0].device).unsqueeze(0)
#         x_pos_emb = self.pos_emb(x_pos)

#         x_cat = x_cat + x_pos_emb
#         attn_mask = self.get_causal_mask(n_iters, n_iters, x_cat.device)

#         for idx, block in enumerate(self.blocks):
#             x_cat, _ = block(x_cat,
#                         attn_mask=attn_mask,
#                         positions=None,
#                         return_kv_cache=False)
        
#         # Reshape back to (B, T, C)
#         output = x_cat[:, -1, :].reshape(B, T, D)
#         output = self.rms_out(output)
#         return output, None
    
#     def forward(self, x: List[Tensor], kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None):
#         B, T, D = x[-1].size()
#         past_iters = 0 if kv_caches is None else kv_caches[0][0].size(2)
#         n_iters = len(x)
#         new_iters = n_iters - past_iters
#         x_cat = torch.stack(x[-new_iters:], dim=2) # (B, T, new_iters, D)
#         x_cat = x_cat.reshape(B*T, -1, D)
#         x_pos = torch.arange(past_iters, n_iters, device=x[0].device).unsqueeze(0)
#         x_pos_emb = self.pos_emb(x_pos)

#         x_cat = x_cat + x_pos_emb

#         attn_mask = self.get_causal_mask(new_iters, n_iters, x_cat.device)
#         # NOTE: In incremental setting, even if attn_mask is None, (full non-causal attention)
#         # The fact that previous states are cached and don't access the future states means that
#         # the model is still causal.
#         # attn_mask: Optional[torch.Tensor] = None


#         updated_kv_caches: List[Tuple[Tensor, Tensor]] = []
#         for idx, block in enumerate(self.blocks):
#             x_cat, kv_cache = block(
#                                 x_cat,
#                                 attn_mask=attn_mask,
#                                 positions=None,
#                                 kv_cache=kv_caches[idx] if kv_caches is not None else None,
#                                 return_kv_cache=True)
            
#             if kv_cache is not None:
#                 updated_kv_caches.append(kv_cache)

#         # Reshape back to (B, T, C)
#         output = x_cat[:, -1, :].reshape(B, T, D)
#         output = self.rms_out(output)

#         return output, updated_kv_caches


class Interpreter(nn.Module):
    def __init__(self, config: REPLConfig) -> None:
        super().__init__()
        self.config = config

        rope_2d = Rope2D(config.n_dim // config.n_head,
                        max_height=config.max_grid_height,
                        max_width=config.max_grid_width)
        self.blocks = nn.ModuleList([TransformerBlock(config, rope=rope_2d) for _ in range(config.n_layer)])
        self.rms_out = RMSNorm(config.n_dim)


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

        x = self.rms_out(x)
        return x, loop_kv_caches




class StateAggregatorRNN(nn.Module):
    def __init__(self, config: REPLConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.n_dim
        self.rnn = nn.GRU(
            input_size=config.n_dim,
            hidden_size=config.n_dim,
            num_layers=1,
            batch_first=True,
        )
        self.rms_out = RMSNorm(config.n_dim)


    def forward(self, x_current: Tensor, h_prev: Optional[Tensor] = None):
        """
        Args:
            x_current (Tensor): Current state tensor of shape (B, T, D).
            h_prev (Tensor, optional): Previous hidden state of shape (B, T, D). Defaults to None.
        Returns:
            output (Tensor): Output tensor of shape (B, T, D).
            h_new (Tensor): New hidden state of shape (B, T, D).
        """
        B, T, D = x_current.shape
        x_input = x_current.view(B * T, 1, D)  # Reshape to (B*T, seq_len=1, D)
        if h_prev is not None:
            h_prev = h_prev.unsqueeze(0).view(1, B * T, self.hidden_size)
        else:
            h_prev = torch.zeros(1, B * T, self.hidden_size, device=x_current.device)

        # Process the current state with the previous hidden state
        output, h_new = self.rnn(x_input, h_prev)  # output: (B*T, 1, D), h_new: (n_layers, B*T, D)

        # Reshape the output back to (B, T, D)
        output = output[:, -1, :].view(B, T, D)
        output = self.rms_out(output)

        # Reshape h_new back to (n_layers, B*T, D) -> (B, T, D)
        h_new = h_new.view(1, B, T, self.hidden_size).squeeze(0)

        return output, h_new


class REPL(nn.Module):
    def __init__(self,
                config: REPLConfig):
        super().__init__()
        self.config = config
        self.n_dim = config.n_dim
        self.n_layer = config.n_layer
        self.num_iters = config.n_iter
        self.pnorm = config.pnorm
        self.PAD_IDX = config.pad_idx

        self.pte = nn.Sequential(
            nn.Embedding(config.prog_vocab_size, config.n_embd),
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

        # color + array + program + inp_grid + out_grid
        self.type_emb = nn.Embedding(5, config.n_dim)

        dummy_idx = torch.ones((1, 1), dtype=torch.long, requires_grad=False)
        self.register_buffer('prog_type_idx', (0 * dummy_idx.clone()))
        self.register_buffer('color_type_idx', (1 * dummy_idx.clone()))
        self.register_buffer('tform_type_idx', (2 * dummy_idx.clone()))
        self.register_buffer('inp_grid_type_idx', (3 * dummy_idx.clone()))
        self.register_buffer('out_grid_type_idx', (4 * dummy_idx.clone()))

        self.interpreter = Interpreter(config)
        self.state_agg = StateAggregatorRNN(config)

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
        pca_indices = torch.full((x.grid.size(0), 3, 2), -1, dtype=grid_indices.dtype, device=x.grid.device)

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
            return_cache: bool = False
        )-> Tuple[List[Tensor], Optional[Tuple[List[List[Tuple[Tensor, Tensor]]], Tensor, Tensor]]]:

        if y is None:
            bs = x.grid.size(0)
            y = MODEL_OUTPUT(
                    grid=torch.zeros((bs, 0), dtype=x.grid.dtype, device=x.grid.device),
                    grid_indices=torch.zeros((bs, 0, 2), dtype=x.grid_indices.dtype, device=x.grid.device),
                    target_grid=None)
                                                 
        enc_inp, enc_valid_mask, enc_indices = self.contruct_encoder_input(x)
        dec_inp, dec_valid_mask, dec_indices = self.contruct_decoder_input(y)

        attn_mask = create_enc_dec_mask(enc_valid_mask, dec_valid_mask).unsqueeze(1)

        dec_start_idx = enc_inp.size(1)

        enc_dec_inp = torch.cat([enc_inp, dec_inp], dim=1)
        enc_dec_indices = torch.cat([enc_indices, dec_indices], dim=1)

        # logits = []
        # iter_states = []

        iter_outs = []

        updated_kv_cache: List[List[Tuple[Tensor, Tensor]]] = [] 

        current_state = enc_dec_inp
        # current_state, states_kv_cache = self.state_agg(iter_states, None)
        for i in range(self.num_iters):
            interpreter_out, iter_kv_cache = self.interpreter(
                                            x=current_state,
                                            attn_mask=attn_mask,
                                            positions=enc_dec_indices,
                                            kv_cache=None,
                                            return_kv_caches=return_cache)
            
            agg_out, current_state = self.state_agg(interpreter_out, current_state)
            iter_outs.append(agg_out)
            # current_state, states_kv_cache = self.state_agg(iter_states, states_kv_cache)
            # logits_i = self.lm_head(current_state[:, dec_start_idx:, :])
            # logits.append(logits_i)
            if iter_kv_cache is not None:
                updated_kv_cache.append(iter_kv_cache)

        iter_outs = torch.stack(iter_outs, dim=0)
        logits = self.lm_head(iter_outs[:, :, dec_start_idx:, :])

        cache = (updated_kv_cache, enc_valid_mask, dec_valid_mask) if return_cache else None
        return logits, cache


    @torch.jit.export
    def forward_inc(self,
            next_y: MODEL_OUTPUT, 
            cache: Tuple[List[List[Tuple[Tensor, Tensor]]], Tensor, Tensor],
        ) -> Tuple[List[Tensor], Tuple[List[List[Tuple[Tensor, Tensor]]], Tensor, Tensor]]:
                   
        dec_inp, dec_valid_mask, dec_indices = self.contruct_decoder_input(next_y)
        seq_len = dec_inp.size(1)
        past_kv_cache, past_enc_valid_mask, past_dec_valid_mask = cache

        dec_valid_mask = torch.cat([past_dec_valid_mask, dec_valid_mask], dim=1)

        attn_mask = create_enc_dec_mask(past_enc_valid_mask, dec_valid_mask).unsqueeze(1)

        # Need to strip the attn_mask to match the size of decoder input
        attn_mask = attn_mask[:, :, -seq_len:, :]

        logits = []
        iter_states = [dec_inp]

        updated_kv_cache: List[List[Tuple[Tensor, Tensor]]] = []
        current_state, states_kv_cache = self.state_agg(iter_states, None)

        for i in range(self.num_iters):
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
            updated_kv_cache.append(iter_kv_cache)

        cache = (updated_kv_cache, past_enc_valid_mask, dec_valid_mask)
        return logits, cache

    
    @torch.jit.export
    def greedy_search(self, 
            prog_idx: int,
            input_grid: List[int],
            input_indices: List[Tuple[int, int]],
            color_perm_idx: int = 0,
            array_tform_idx: int = 0,
            max_length: int = 30*30,
            bos_idx: int = GridTokenizer().BOS_IDX,
            eos_idx: int = GridTokenizer().EOS_IDX,
            new_row_idx: int = GridTokenizer().NEW_ROW_IDX,
            max_grid_height: int = 60,
            max_grid_width: int = 60
            )-> Tuple[List[int], float]:

        torch.set_grad_enabled(False)
        device = self.type_emb.weight.device  # Get the device from the embedding layer (assuming it's available)

        # Convert input_indices to a list of lists to make torchscript happy
        input_indices_list = [list(t) for t in input_indices]

        x = MODEL_INPUT(
            program=torch.tensor([[prog_idx]], dtype=torch.long, device=device),
            color_permutation=torch.tensor([[color_perm_idx]], dtype=torch.long, device=device),
            array_transform=torch.tensor([[array_tform_idx]], dtype=torch.long, device=device),
            grid=torch.tensor([input_grid], dtype=torch.long, device=device),
            grid_indices=torch.tensor([input_indices_list], dtype=torch.long, device=device),  # Fixed line
            meta=None
        )

        _, cache = self.forward(
                x=x,
                y=None,
                return_cache=True
        )

        assert cache is not None, "Cache must be returned for greedy search"

        # First token is BOS token always to start the generation
        last_token = bos_idx
        last_token_r, last_token_c = 0, 0
        # # Annotate the empty tensor for TorchScript
        output_sequence = torch.empty(0, dtype=torch.long, device=device)  # Shape: (seq_len,)
        output_log_prob = 0.0

        max_r, max_c = max_grid_height-1, max_grid_width-1


        for t in range(max_length):
            next_y = MODEL_OUTPUT(
                grid=torch.tensor([[last_token]], dtype=torch.long, device=device),  # Shape: (1, 1),
                grid_indices=torch.tensor([[[last_token_r, last_token_c]]], dtype=torch.long, device=device),  # Shape: (1, 1, 2)
                target_grid=None
            )

            logits_iters, cache = self.forward_inc(
                next_y=next_y,
                cache=cache
            )

            # Get the logits from the last iteration
            logits = logits_iters[-1]  

            # Get the logits for the predicted token
            next_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
            next_log_probs = F.log_softmax(next_logits, dim=-1)


            # Select the token with the highest probability
            top_token = torch.argmax(next_logits, dim=-1)  # Shape: (1,)
            top_log_prob = next_log_probs[0, top_token].item()
            output_log_prob += top_log_prob

            # Append the new token to the output sequence
            output_sequence = torch.cat([output_sequence, top_token])

            token_token_idx = top_token.item()

            if token_token_idx == new_row_idx:
                last_token_r = min(last_token_r + 1, max_r)
                last_token_c = 0
            else:
                last_token_c = min(last_token_c + 1, max_c)
            # If EOS token is generated, stop
            if token_token_idx == eos_idx:
                break

            last_token = token_token_idx

        output_list: List[int] = output_sequence.tolist()  # Use .tolist() now since it's supported in TorchScript
        torch.set_grad_enabled(True)
        return output_list, output_log_prob
    
    @torch.jit.ignore
    def get_optimizer(
            self, 
            model_lr,
            prog_lr,
            model_wd,
            prog_wd=0.0,
            prog_l1=0.0,
            device_type=None
        ):

        # Freeze model params if model_lr is 0, needed for finetuning
        if model_lr == 0.0:
            logger.warning("Freezing model parameters. Only Program embedding parameters will be trained.")
            logger.warning("This setting should only be used for training without resuming/forkin (with optimizer state load disabled)")

            for n, p in self.named_parameters():
                if 'pte.0' not in n:
                    p.requires_grad = False

        # Separate the embedding parameters
        program_params = [p for n, p in self.named_parameters() if 'pte.0' in n and p.requires_grad]
        model_params = [p for n, p in self.named_parameters() if 'pte.0' not in n and p.requires_grad]

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in model_params if p.dim() >= 2]
        nodecay_params = [p for p in model_params if p.dim() < 2]
        optim_groups = [
            {'params': program_params,
                'lr': prog_lr,
                'weight_decay': prog_wd,
                'l1_coeff': prog_l1},
            {'params': decay_params,
                'lr': model_lr,
                'weight_decay': model_wd,
                'l1_coeff': 0.0},
            {'params': nodecay_params,
                'lr': model_lr,
                'weight_decay': 0.0,
                'l1_coeff': 0.0}]
        # Create AdamW optimizer and use the fused version if it is available
        use_fused = False
        if torch.cuda.is_available() and (device_type is None or device_type == 'cuda'):
            use_fused = True
            print(f"Using fused AdamW: {use_fused}")
            
        optimizer = LazyAdamW(optim_groups, lr=model_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


    def load_state_dict(self, state_dict, strict: bool=True, assign: bool=False):

        if strict:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        pte_w_key = 'pte.0.weight'
        src_sd = self.state_dict()
        trg_sd = state_dict
        
        if self.state_dict()[pte_w_key].shape != state_dict[pte_w_key].shape:
            logger.warning(f"WARNING: Skipping loading Program Embeddings as they differ in shape in the target model. ")
            trg_sd = {k: v for k, v in state_dict.items() if k != pte_w_key}
            trg_sd[pte_w_key] = self.state_dict()[pte_w_key]


        def get_num_blocks(state_dict, prefix):
            max_block_id = 0

            for key in state_dict.keys():
                # Use regex to match the pattern
                match = re.match(rf'{prefix}\.blocks\.(\d+)\.', key)
                if match:
                    block_number = int(match.group(1))
                    max_block_id = max(max_block_id, block_number)

            return max_block_id + 1

        @torch.no_grad()
        def copy_(prefix, idx_mapping=None, src_prefix=None):
            for name, t in trg_sd.items():
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    src_name = src_prefix + suffix if src_prefix is not None else name
                    s = src_sd[src_name]
                    trg_ptr_b4 = t.data_ptr()
                    if idx_mapping is None:
                        t.data.copy_(s)
                    else:
                        for trg_idx, src_idx in idx_mapping.items():
                            t.data[trg_idx].copy_(s.data[src_idx])

                    trg_ptr_after = t.data_ptr()
                    assert trg_ptr_b4 == trg_ptr_after, f"Data pointer changed for {prefix}"

        missing_keys, unexpected_keys = super().load_state_dict(trg_sd, strict=strict, assign=assign)

        for k in missing_keys + unexpected_keys:
            assert k.startswith('interpreter.') or k.startswith('state_agg.'), f"Unexpected key: {k}"

        def deal_with_blocks(prefix):
            num_trg_blocks = get_num_blocks(self.state_dict(), prefix)
            num_src_blocks = get_num_blocks(trg_sd, prefix)
            num_params_per_block = 7

            prefix_missing_keys = [k for k in missing_keys if k.startswith(f'{prefix}.blocks')]    
            prefix_unexpected_keys = [k for k in unexpected_keys if k.startswith(f'{prefix}.blocks')]

            if num_trg_blocks > num_src_blocks:
                logger.warning(f"WARNING: Target model has more {prefix} blocks than the source model")
                assert len(prefix_unexpected_keys) == 0, f"Missing keys: {prefix_unexpected_keys}"
                assert len(prefix_missing_keys) == num_params_per_block * (num_trg_blocks - num_src_blocks), f"Unexpected keys: {prefix_missing_keys}"

                # This is the case where we do block expansion
                for trg_block_id in range(num_src_blocks, num_trg_blocks):
                    trg_block_key = f'{prefix}.blocks.{trg_block_id}'
                    src_block_idx = max(trg_block_id - (num_trg_blocks - num_src_blocks), 0)
                    src_block_key = f'{prefix}.blocks.{src_block_idx}'
                    copy_(f'{trg_block_key}.', src_prefix=f'{src_block_key}.')

                    with torch.no_grad():
                        module = self.interpreter if prefix == 'interpreter' else self.state_agg
                        # Set the SwiGLUFFN output multiplication close to zero
                        module.blocks[trg_block_id].normed_mlp[1].w3.weight.normal_(mean=0, std=0.0001)
                        # Set the value project close to zero!
                        module.blocks[trg_block_id].attn.c_attn.weight[self.n_dim * 2:, :].normal_(mean=0, std=0.0001)

                    logger.warning(f"WARNING: Copying {prefix} block idx: {trg_block_id} from the block idx: {src_block_idx} of the source model and made it identity block")

            elif num_trg_blocks < num_src_blocks:
                logger.warning(f"WARNING: Target model has less {prefix} blocks than the source model. Copy only the first {num_trg_blocks} blocks")
                assert len(prefix_missing_keys) == 0, f"Unexpected keys: {prefix_missing_keys}"
                assert len(prefix_unexpected_keys) == num_params_per_block * (num_src_blocks - num_trg_blocks), f"Missing keys: {prefix_unexpected_keys}"

        deal_with_blocks('interpreter')
        deal_with_blocks('state_agg')


    def load_prog_embeddings(self, trg_token2idx, src_state_dict, src_token2idx):
        src_sd = src_state_dict
        trg_sd = self.state_dict()

        @torch.no_grad()
        def copy_(prefix, idx_mapping=None, src_prefix=None):
            for name, t in trg_sd.items():
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    src_name = src_prefix + suffix if src_prefix is not None else name
                    s = src_sd[src_name]
                    trg_ptr_b4 = t.data_ptr()
                    if idx_mapping is None:
                        t.data.copy_(s)
                    else:
                        for trg_idx, src_idx in idx_mapping.items():
                            t.data[trg_idx].copy_(s.data[src_idx])

                    trg_ptr_after = t.data_ptr()
                    assert trg_ptr_b4 == trg_ptr_after, f"Data pointer changed for {prefix}"

        @torch.no_grad()
        def copy_embedding_weights(key, trg_token2idx, src_token2idx):
            common_tokens = set(trg_token2idx.keys()).intersection(set(src_token2idx.keys()))

            if set(trg_token2idx.keys()) != set(src_token2idx.keys()):
                logger.warning(f"WARNING: Tokens for {key} are not the same. Source has {len(src_token2idx)} tokens and target has {len(trg_token2idx)} tokens. Copying {len(common_tokens)} common tokens.")

            token_idx_mapping = {trg_token2idx[token]: src_token2idx[token] for token in common_tokens}
            copy_(key, token_idx_mapping)


        copy_embedding_weights('pte.0.', trg_token2idx, src_token2idx)