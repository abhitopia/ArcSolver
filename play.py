#%%
from collections import namedtuple
from typing import List, Optional, Tuple
from torch.nn import functional as F

import numpy as np
import torch
from torch import Tensor
from src.dataset1 import ArcExamplesDataset
from src.interpreter import RMSNorm, RotaryPositionalEmbeddings, SwiGLUFFN
from src.task1 import TRAIN_COLLECTION, ColorPermutation, ArrayTransform, Example
from src.tokenizer import ArcTokenizer, ArrayTransformTokenizer, ColorPermutationTokenizer, GridTokenizer
from src.tokenizer import ArcTokenizer, MODEL_OUTPUT, MODEL_INPUT

# %%
train_examples = TRAIN_COLLECTION.train_examples
test_examples = TRAIN_COLLECTION.test_examples
arc_tokenizer = ArcTokenizer()
arc_tokenizer.build_program_tokenizer(train_examples)
# %%
len(train_examples), len(test_examples)
# %%
ds = ArcExamplesDataset(train_examples, arc_tokenizer)
dl = ds.get_dataloader(32000)
for batch in dl:
    x, y = batch
    print(x.input.shape, y.output.shape)
    break
x, y = batch
# %%
from dataclasses import dataclass
from src.utils import is_power_of_two, get_logger, gather_4d_tensor_along_zero_dim
import torch.nn as nn


@dataclass
class InterpreterConfig:
    prog_vocab_size: int # number of program tokens
    n_dim: int  # dimension of the model
    n_embd: int  # embedding dimension
    n_head: int # number of heads within each self-attention block
    n_layer: int  # number of transformer blocks / layers
    pnorm: float = None # target L2 norm for embedding vectors
    max_seq_len: int = 2048 # max sequence length
    grid_vocab_size: int = len(GridTokenizer()) # number of array element tokens (one extra for niceness)
    perm_vocab_size: int = len(ColorPermutationTokenizer())
    tform_vocab_size: int = len(ArrayTransformTokenizer())
    dropout: float = 0.0 # dropout probability

    def __post_init__(self):
        # assert is_power_of_two(self.prog_vocab_size), "Program vocab size must be a power of 2"
        # assert is_power_of_two(self.grid_vocab_size), "Grid vocab size must be a power of 2"
        # assert is_power_of_two(self.n_dim), "Model dimension must be a power of 2"

        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        if self.pnorm is not None:
            assert 0.0 < self.pnorm, "p-norm must be greater than 0"
        
        head_dim = self.n_dim // self.n_head
        assert head_dim % 2 == 0, "Head dimension must be even"


    def to_dict(self):
        return {
            'prog_vocab_size': self.prog_vocab_size,
            'grid_vocab_size': self.grid_vocab_size,
            'perm_vocab_size': self.perm_vocab_size,
            'tform_vocab_size': self.tform_vocab_size,
            'n_dim': self.n_dim,
            'n_embd': self.n_embd,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'max_seq_len': self.max_seq_len,
            'pnorm': self.pnorm,
            'dropout': self.dropout
        }
    
    @staticmethod
    def from_dict(data: dict) -> "InterpreterConfig":
        return InterpreterConfig(**data)


class SelfAttention(nn.Module):
    def __init__(self, config: InterpreterConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()
        self.config = config
        self.rope = rope
        assert config.n_dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)

        self.c_attn_q = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.c_attn_k = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.c_attn_v = nn.Linear(config.n_dim, config.n_dim, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.c_proj.RESCALE_RESIDUAL = 1

        # regularization
        self.n_head = config.n_head
        self.n_dim = config.n_dim
        self.dropout = config.dropout


    # def forward(self, x, attn_mask=None, kv_cache=None, return_kv_cache=False):

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
            attn_mask: Optional[Tensor] = None, 
            kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # x: (B, T, C)
        B, qT, C = q.size()
        kT = k.size(1)

        # qkv: (B, T, 3 * C)
        # qkv = self.c_attn(x)
        # q, k, v = qkv.split(self.n_dim, dim=2)
        q = self.c_attn_q(q)
        k = self.c_attn_k(k)
        v = self.c_attn_v(v)

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, qT, self.n_head, C // self.n_head)  # (B, T, n_head, head_dim)
        k = k.view(B, kT, self.n_head, C // self.n_head)
        v = v.view(B, kT, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)

        # If kv_cache is present, concatenate past keys and values BEFORE applying RoPE
        if kv_cache is not None and torch.jit.isinstance(kv_cache, Tuple[Tensor, Tensor]):
            past_k, past_v = kv_cache  # K: (B, T_past, n_head, head_dim), V: (B, n_head, T_past, head_dim)
            k = torch.cat([past_k, k], dim=1)  # Concatenate along sequence length dimension
            v = torch.cat([past_v, v], dim=2)

        # Update new_kv_cache
        new_kv_cache: Optional[Tuple[Tensor, Tensor]] = (k, v) if return_kv_cache else None


        # Generate position indices for the concatenated sequence
        total_seq_len = k.size(1)  # k now contains both past and current
        position_ids = torch.arange(total_seq_len, dtype=torch.long, device=q.device)
        position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(B, 1, total_seq_len)

        # Apply RoPE to q and k before transposing for attention
        # For q, we use positions corresponding to the current tokens (last T positions)
        q_positions = position_ids[:, :, -qT:]  # Shape: (B, 1, T)
        q = self.rope(q, input_pos=q_positions)

        # For k, we use positions for the entire sequence (past + current)
        k_positions = position_ids  # Shape: (B, 1, total_seq_len)
        k = self.rope(k, input_pos=k_positions)

        # Now transpose q and k for attention computation
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_head, total_seq_len, head_dim)
        
        # Compute attention
        dropout_p = self.dropout if self.training else 0.0

        # default scale (1/sqrt(D)) is applied inside scaled_dot_product_attention
        attn_output = F.scaled_dot_product_attention(q, k, v, 
                                                    attn_mask=attn_mask,
                                                    # is_causal=attn_mask is None, # default to causal if attn_mask is None
                                                    dropout_p=dropout_p)

        # attn_output: (B, n_head, T, head_dim)
        # Reshape back to (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, qT, C)

        # Output projection
        y = self.c_proj(attn_output)

        return y, new_kv_cache



class EncoderBlock(nn.Module):
    def __init__(self, config: InterpreterConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.rmsnorm = RMSNorm(config.n_dim)
        self.attn = SelfAttention(config, rope)
        self.normed_mlp = nn.Sequential(
                            RMSNorm(config.n_dim),
                            SwiGLUFFN(config.n_dim, 4 * config.n_dim))

    def forward(self,
            x: Tensor,
            attn_mask: Optional[Tensor] = None, 
            kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:

        x = self.rmsnorm(x)
        attn_output, new_kv_cache = self.attn(x, x, x, attn_mask=attn_mask, kv_cache=kv_cache, return_kv_cache=return_kv_cache)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.normed_mlp(x))
        return x, new_kv_cache



class Encoder(nn.Module):
    def __init__(self, config: InterpreterConfig):
        super().__init__()

        self.config = config
        self.n_dim = config.n_dim
        self.n_layer = config.n_layer
        self.max_seq_len = config.max_seq_len
        rope = RotaryPositionalEmbeddings(config.n_dim // config.n_head, config.max_seq_len)
        self.enc_blocks = nn.ModuleList([EncoderBlock(config, rope=rope) for _ in range(config.n_layer)])

        self.rms_out = RMSNorm(config.n_dim)

    def forward(self, x: torch.Tensor, enc_mask: Optional[torch.Tensor] = None, return_kv_caches: bool = False):
        updated_kv_caches: List[Tuple[Tensor, Tensor]] = []
        for block in self.enc_blocks:
            x, kv_cache = block(x, attn_mask=enc_mask, return_kv_cache=False)
            if return_kv_caches and kv_cache is not None:
                updated_kv_caches.append(kv_cache)

        # This is important since, we don't apply normalisation when 
        # this is passed to the decoder
        x = self.rms_out(x)
        return x, updated_kv_caches
    

class DecoderBlock(nn.Module):
    def __init__(self, config: InterpreterConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.rmsnorm_1 = RMSNorm(config.n_dim)
        self.rmsnorm_2 = RMSNorm(config.n_dim)

        self.attn = SelfAttention(config, rope)
        self.normed_mlp = nn.Sequential(
                            RMSNorm(config.n_dim),
                            SwiGLUFFN(config.n_dim, 4 * config.n_dim))

    def forward(self,
            enc_out: Tensor,
            x: Tensor,
            enc_mask: Optional[Tensor] = None,
            dec_mask: Optional[Tensor] = None, 
            kv_cache: Optional[Tuple[Tensor, Tensor]] = None, 
            return_kv_cache: bool = False) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:

        x = self.rmsnorm_1(x)
        attn_output, new_kv_cache = self.attn(x, x, x, attn_mask=dec_mask, kv_cache=kv_cache, return_kv_cache=return_kv_cache)
        x = x + self.dropout(attn_output)


        x = self.rmsnorm_2(x)
        # Notice that enc_out is passed without normalisation
        # This is because the normalisation is applied in the encoder
        attn_output, new_kv_cache = self.attn(x, enc_out, enc_out, attn_mask=enc_mask, kv_cache=kv_cache, return_kv_cache=return_kv_cache)

        # MLP block
        x = x + self.dropout(self.normed_mlp(x))
        return x, new_kv_cache


class Decoder(nn.Module):
    def __init__(self, config: InterpreterConfig):
        super().__init__()

        self.config = config
        self.n_dim = config.n_dim
        self.n_layer = config.n_layer
        self.max_seq_len = config.max_seq_len
        rope = RotaryPositionalEmbeddings(config.n_dim // config.n_head, config.max_seq_len)
        self.dec_blocks = nn.ModuleList([DecoderBlock(config, rope=rope) for _ in range(config.n_layer)])


    def forward(self, 
            enc_out: Tensor, 
            x: torch.Tensor, 
            enc_mask: Optional[torch.Tensor] = None,   
            dec_mask: Optional[Tensor] = None, 
            return_kv_caches: bool = False
        ):

        updated_kv_caches: List[Tuple[Tensor, Tensor]] = []
        for block in self.dec_blocks:
            x, kv_cache = block(enc_out, x, enc_mask=enc_mask, dec_mask=dec_mask, return_kv_cache=False)
            if return_kv_caches and kv_cache is not None:
                updated_kv_caches.append(kv_cache)
        return x, updated_kv_caches
    

class LoopEncoder(nn.Module):
    def __init__(self, config: InterpreterConfig):
        super().__init__()
        self.config = config
        self.n_dim = config.n_dim
        self.n_layer = config.n_layer
        self.max_seq_len = config.max_seq_len
        rope = RotaryPositionalEmbeddings(config.n_dim // config.n_head, config.max_seq_len)
        self.enc_blocks = nn.ModuleList([EncoderBlock(config, rope=rope) for _ in range(config.n_layer)])
        self.rms_out = RMSNorm(config.n_dim)

    def forward(self, x: List[torch.Tensor]):
        B, T, C = x[0].size()

        x_reshape = [i.reshape(B*T, C) for i in x]
        print("Loops", [i.shape for i in x_reshape])

        x_cat = torch.stack(x_reshape, dim=0).permute(1, 0, 2)

        print("x_cat", x_cat.shape)

        for block in self.enc_blocks:
            x_cat, _ = block(x_cat, attn_mask=None, return_kv_cache=False)
        
        print("x_cat", x_cat.shape)

        output = x_cat[:, -1, :].reshape(B, T, C)
        print("output", output.shape)

        return output


class Interpreter(nn.Module):
    def __init__(self,
                config: InterpreterConfig,
                pad_idx: int = GridTokenizer().PAD_IDX):
        super().__init__()
        self.config = config
        self.n_dim = config.n_dim
        self.n_layer = config.n_layer
        self.max_seq_len = config.max_seq_len
        self.pnorm = config.pnorm

        self.PAD_IDX = pad_idx

        self.pte = nn.Sequential(
            nn.Embedding(config.prog_vocab_size, config.n_embd, sparse=True),
            nn.Linear(config.n_embd, config.n_dim, bias=False)
        )

        self.gte = nn.Sequential(
            nn.Embedding(config.grid_vocab_size, config.n_embd),
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

        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.loop_encoder = LoopEncoder(config)
        self.loop_decoder = LoopEncoder(config)
        # self.inp_inject = nn.Linear(2*config.n_dim, config.n_dim, bias=False)
        # self.dec_blocks = nn.ModuleList([TransformerBlock(config, rope=rope) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_dim)

        self.lm_head = nn.Linear(config.n_dim, config.grid_vocab_size, bias=False)
        
        self._untrained_block_ids = []
        
        # weight sharing scheme. Transformer++ (Llama architecture does not tie weights)
        # Reference: https://youtu.be/pRM_P6UfdIc?t=1500
        # self.wte.weight = self.lm_head.weight

        # init params
        # self.apply(self._init_weights)
        # self.init_prog_embedding()

    def init_prog_embedding(self):
        if self.pnorm is not None:
            """Initialize embedding vectors with the target L2 norm."""
            with torch.no_grad():
                # Initialize the embedding weights to random values
                nn.init.normal_(self.pte.weight)
                # Normalize each embedding vector to the target L2 norm
                norm = self.pte.weight.norm(p=2, dim=1, keepdim=True)
                self.pte.weight = nn.Parameter(self.pte.weight * (self.pnorm / norm))


    @staticmethod
    def enc_mask(inp_grid, program_prefix=3, pad_idx=13):
        T = inp_grid.size(1) + program_prefix
        device = inp_grid.device
        inp_lens = (inp_grid != pad_idx).sum(1).unsqueeze(1) + program_prefix
        idx = torch.arange(T, device=device)
        mask = (idx < inp_lens).unsqueeze(1).expand(-1, T, -1)
        mask = mask & mask.transpose(1, 2)
        return mask.unsqueeze(1)
    
    @staticmethod
    def dec_mask(output_grid):
        qT = output_grid.size(1)
        bs = output_grid.size(0)
        device = output_grid.device
        causal_mask = torch.ones(bs, qT, qT, dtype=torch.bool, device=device).tril(diagonal=0).unsqueeze(1)
        return causal_mask
    
    @staticmethod
    def dec_enc_mask(inp_grid, output_grid, program_prefix=3, pad_idx=13):
        kT = inp_grid.size(1) + program_prefix
        qT = output_grid.size(1)
        device = inp_grid.device
        inp_lens = (inp_grid != pad_idx).sum(1).unsqueeze(1).unsqueeze(2) + program_prefix
        idx = torch.arange(kT, device=device)
        mask = (idx < inp_lens).expand(-1, qT, -1)
        return mask.unsqueeze(1)

    def forward(self, x: torch.Tensor, n_loops: int = 1):
        color_permutation = x.color_permutation
        array_transform = x.array_transform
        program = x.program
        input_grid = x.input
        output_grid = x.causal_output
        enc_mask = self.enc_mask(input_grid, program_prefix=3, pad_idx=self.PAD_IDX)
        dec_mask = self.dec_mask(output_grid)
        dec_enc_mask = self.dec_enc_mask(input_grid, output_grid, program_prefix=3, pad_idx=self.PAD_IDX)

        enc_mask = None
        # dec_enc_mask = None
        # dec_mask = None

        program = self.pte(program)
        color_permutation = self.cte(color_permutation)
        array_transform = self.ate(array_transform)
        inp_grid = self.gte(input_grid)

        enc_inp = torch.cat([program, color_permutation, array_transform, inp_grid], dim=1)
        dec_inp = self.gte(output_grid)

        loop_enc_inps = []
        loop_dec_inps = []
        for _ in range(n_loops):
            enc_inp, enc_kv_cache = self.encoder(enc_inp, enc_mask=enc_mask)
            loop_enc_inps.append(enc_inp)
            enc_inp = self.loop_encoder(loop_enc_inps)

            dec_inp, dec_kv_cache = self.decoder(enc_inp, dec_inp, enc_mask=dec_enc_mask, dec_mask=dec_mask)
            loop_dec_inps.append(dec_inp)
            dec_inp = self.loop_decoder(loop_dec_inps)

        print(enc_inp.shape, dec_inp.shape)
        dec_out = self.ln_f(dec_inp)
        logits = self.lm_head(dec_out)
        return logits
    

config = InterpreterConfig(
    prog_vocab_size=len(arc_tokenizer.program_tokenizer),
    n_dim=256,
    n_embd=256,
    n_head=8,
    n_layer=2,
    max_seq_len=2048,
    pnorm=1.0,
    dropout=0.0
)

interpreter = Interpreter(config)
interpreter(x, 3).shape
# %%


def dec_enc_mask(inp_grid, output_grid, program_prefix=3, pad_idx=13):
    kT = inp_grid.size(1) + program_prefix
    qT = output_grid.size(1)

    inp_lens = (inp_grid != pad_idx).sum(1).unsqueeze(1).unsqueeze(2) + program_prefix
    idx = torch.arange(kT, device=device)
    mask = (idx < inp_lens).expand(-1, qT, -1)
    return mask.unsqueeze(1)



dec_enc_mask(x.input, x.causal_output)[4]
# %%

# bs = 6
kT = 5
qT = 3
device = x.input.device

enc_inp = torch.tensor(
    [[1, 2, 3, 13, 13, 13],
     [1, 2, 3, 4, 13, 13],
     [1, 13, 13, 13, 13, 13]])

dec_inp = torch.ones(3, 8)

dec_enc_mask(enc_inp, dec_inp)[2]
#      k1, k2, k3, k4 k5
# q1 [   ,   ,   ,   ,   ]
# q2 [   ,   ,   ,   ,   ]
# q3 [   ,   ,   ,   ,   ]
# %%

(x.input != 13).sum(1)
# %%

x.causal_output.shape, x.input.shape
# %%
