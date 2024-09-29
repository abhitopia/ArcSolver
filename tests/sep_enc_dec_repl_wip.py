#%%
from collections import namedtuple
from typing import List, Optional, Tuple
from torch.nn import functional as F

import numpy as np
import torch
from torch import Tensor
from src.dataset1 import ArcExamplesDataset
from src.interpreter import RMSNorm, RotaryPositionalEmbeddings, SwiGLUFFN
from src.multilevel_loss import MultiLevelLoss
from src.task1 import TRAIN_COLLECTION, ColorPermutation, ArrayTransform, Example
from src.tokenizer import ArcTokenizer, ArrayTransformTokenizer, ColorPermutationTokenizer, GridTokenizer
from src.tokenizer import ArcTokenizer, MODEL_OUTPUT, MODEL_INPUT
from src.utils import debug_context, debug_print
# %%
train_examples = TRAIN_COLLECTION.train_examples
test_examples = TRAIN_COLLECTION.test_examples
arc_tokenizer = ArcTokenizer()
arc_tokenizer.build_program_tokenizer(train_examples)

ds = ArcExamplesDataset(train_examples, arc_tokenizer)
dl = ds.get_dataloader(32000)

for batch in dl:
    x, y = batch
    print(x.input.shape, y.output.shape)

    non_pad = (x.input != 13).sum()
    total = x.input.numel()
    print(non_pad, total, non_pad/total)
    break
x, y = batch
# %%
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class REPLConfig:
    prog_vocab_size: int # number of program tokens
    n_dim: int  # dimension of the model
    n_embd: int  # embedding dimension
    n_head: int # number of heads within each self-attention block
    n_dec_layer: int  # number of transformer decoder blocks per loop
    n_enc_layer: int  # number of transformer encoder blocks per loop
    n_lecn_layer: int # number of transformer encoder blocks across loops
    n_ldec_layer: int # number of transformer decoder blocks across loops
    pnorm: float = None # target L2 norm for embedding vectors
    max_seq_len: int = 2048 # max sequence length per encoder/decoder
    grid_vocab_size: int = len(GridTokenizer()) # number of array element tokens (one extra for niceness)
    perm_vocab_size: int = len(ColorPermutationTokenizer())
    tform_vocab_size: int = len(ArrayTransformTokenizer())
    dropout: float = 0.0 # dropout probability
    edr: float = 2.0 # exponential error decay rate per loop, 0.0 means uniform dep rate
    mctp: float = 0.4 # minimum correct token percentage for loss computation
    max_loops: int = 20 # maximum number of loops
    loop_rope_base: int = 1_000 # Smaller for higher sensitivity to loop iterations



    def __post_init__(self):
        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        if self.pnorm is not None:
            assert 0.0 < self.pnorm, "p-norm must be greater than 0"
        
        assert self.mctp < 1.0, "Minimum error rate must be less than 1.0"

        assert self.edr >= 0.0, "Error depreciation rate must be greater than 0.0"
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
            'n_dec_layer': self.n_dec_layer,
            'n_enc_layer': self.n_enc_layer,
            'n_lecn_layer': self.n_lecn_layer,
            'n_ldec_layer': self.n_ldec_layer,
            'max_seq_len': self.max_seq_len,
            'pnorm': self.pnorm,
            'dropout': self.dropout,
            'edr': self.edr,
            'mctp': self.mctp,
            'max_loops': self.max_loops,
            'loop_rope_base': self.loop_rope_base
        }
    
    @staticmethod
    def from_dict(data: dict) -> "REPLConfig":
        return REPLConfig(**data)


class SelfAttention(nn.Module):
    def __init__(self, config: REPLConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_dim = config.n_dim
        self.dropout = config.dropout
        self.rope = rope
        assert config.n_dim % config.n_head == 0

        # query, key, value projections
        self.c_attn_q = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.c_attn_k = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.c_attn_v = nn.Linear(config.n_dim, config.n_dim, bias=False)

        # output projection
        self.c_proj = nn.Linear(config.n_dim, config.n_dim, bias=False)


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

        debug_print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, qT, self.n_head, C // self.n_head)  # (B, T, n_head, head_dim)
        k = k.view(B, kT, self.n_head, C // self.n_head)
        v = v.view(B, kT, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)

        # If kv_cache is present, concatenate past keys and values BEFORE applying RoPE
        if kv_cache is not None:
            past_k, past_v = kv_cache  # K: (B, T_past, n_head, head_dim), V: (B, n_head, T_past, head_dim)
            k = torch.cat([past_k, k], dim=1)  # Concatenate along sequence length dimension
            v = torch.cat([past_v, v], dim=2)
            # debug_print(f"Found kv_cache: {k[~torch.isnan(k)].sum()}")
            # debug_print(f"Found kv_cache: {v[~torch.isnan(v)].sum()}")
        else:
            # debug_print(f"No found kv_cache: {k[~torch.isnan(k)].sum()}")
            # debug_print(f"No found kv_cache: {v[~torch.isnan(v)].sum()}")
            pass

        # Update new_kv_cache
        new_kv_cache: Optional[Tuple[Tensor, Tensor]] = (k, v) if return_kv_cache else None


        # # TODO: Below logic is utterly wrong for encoder-decoder architecture

        # # Generate position indices for the concatenated sequence
        total_seq_len = k.size(1)  # k now contains both past and current
        position_ids = torch.arange(total_seq_len, dtype=torch.long, device=q.device)
        position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(B, 1, total_seq_len)

        # # Apply RoPE to q and k before transposing for attention
        # # For q, we use positions corresponding to the current tokens (last T positions)
        q_positions = position_ids[:, :, -qT:]  # Shape: (B, 1, T)
        q = self.rope(q, q_positions)

        # # For k, we use positions for the entire sequence (past + current)
        k_positions = position_ids  # Shape: (B, 1, total_seq_len)
        k = self.rope(k, k_positions)
        # debug_print(f"RoPE: {k[~torch.isnan(k)].sum()}")

        # Now transpose q and k for attention computation
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_head, total_seq_len, head_dim)
        
        # Compute attention
        dropout_p = self.dropout if self.training else 0.0

        # debug_print(f"Attention input: {q[~torch.isnan(q)].sum(), k[~torch.isnan(k)].sum(), v[~torch.isnan(v)].sum()}")

        # last_q = q[:, :, -1, :]
        # debug_print(f"Attention inp Last Q: {last_q[~torch.isnan(last_q)].sum()}")
        # default scale (1/sqrt(D)) is applied inside scaled_dot_product_attention
        # debug_print(f"Attention mask: {attn_mask.shape if attn_mask is not None else None}")

        # Default Causal Mask is broken :(, need to create my own
        attn_output = F.scaled_dot_product_attention(q, k, v, 
                                                    attn_mask=attn_mask,
                                                    is_causal=attn_mask is None, # default to causal if attn_mask is None
                                                    dropout_p=dropout_p)

        last_output = attn_output[:, :, -1, :].unsqueeze(2)
        debug_print(f"Last output shape {last_output.shape}")
        debug_print(f"Attention output (after attention): {last_output[~torch.isnan(last_output)].sum()}")
        if q.size(2) == 2:
            last_q = q[:, :, -1, :].unsqueeze(2)
            debug_print(f"Inside: Attention inp Last Q: {last_q[~torch.isnan(last_q)].sum()}")
            attn_output_last = F.scaled_dot_product_attention(last_q, k, v, 
                                                attn_mask=attn_mask,
                                                is_causal=attn_mask is None, # default to causal if attn_mask is None
                                                dropout_p=dropout_p)
            debug_print(f"Attn output shape {attn_output_last.shape}")
            debug_print(f"Inside: Attention output: {attn_output_last[~torch.isnan(attn_output_last)].sum()}")

        # attn_output: (B, n_head, T, head_dim)
        # Reshape back to (B, T, C)

        # debug_print(f"Output shape before: {attn_output.shape}, {B, qT, C, kT}")
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, qT, C)
        # debug_print(f"Output shape after: {attn_output.shape}")

        last_output = attn_output[:, -1, :]
        # debug_print(f"Attention output (before proj): {last_output[~torch.isnan(last_output)].sum()}")
        # Output projection
        y = self.c_proj(attn_output) # (B, T, C)
        # debug_print(f"y shape: {y.shape}")
        last_output = y[:, -1, :]
        # debug_print(f"Attention output (after proj): {last_output[~torch.isnan(last_output)].sum()}")
        return y, new_kv_cache



class EncoderBlock(nn.Module):
    def __init__(self, config: REPLConfig, rope: RotaryPositionalEmbeddings):
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
    def __init__(self, config: REPLConfig):
        super().__init__()

        self.config = config
        self.n_dim = config.n_dim
        self.max_seq_len = config.max_seq_len
        rope = RotaryPositionalEmbeddings(config.n_dim // config.n_head, config.max_seq_len)
        self.enc_blocks = nn.ModuleList([EncoderBlock(config, rope=rope) for _ in range(config.n_enc_layer)])

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
    def __init__(self, config: REPLConfig, rope: RotaryPositionalEmbeddings):
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
    def __init__(self, config: REPLConfig):
        super().__init__()

        self.config = config
        self.n_dim = config.n_dim
        self.max_seq_len = config.max_seq_len
        rope = RotaryPositionalEmbeddings(config.n_dim // config.n_head, config.max_seq_len)
        self.dec_blocks = nn.ModuleList([DecoderBlock(config, rope=rope) for _ in range(config.n_dec_layer)])

        self.rms_out = RMSNorm(config.n_dim)

    def forward(self, 
            enc_out: Tensor, 
            x: torch.Tensor, 
            enc_mask: Optional[torch.Tensor] = None,   
            dec_mask: Optional[Tensor] = None,
            kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
            return_kv_caches: bool = False
        ):

        updated_kv_caches: List[Tuple[Tensor, Tensor]] = []
        for block in self.dec_blocks:
            x, kv_cache = block(enc_out, x, enc_mask=enc_mask, dec_mask=dec_mask, return_kv_cache=False)
            if return_kv_caches and kv_cache is not None:
                updated_kv_caches.append(kv_cache)

        x = self.rms_out(x)
        return x, updated_kv_caches
    

class IterativeEncoder(nn.Module):
    def __init__(self, config: REPLConfig, n_layer):
        super().__init__()
        self.config = config
        self.n_dim = config.n_dim
        self.n_layer = n_layer
        self.max_seq_len = config.max_seq_len
        rope = RotaryPositionalEmbeddings(
                                    config.n_dim // config.n_head,
                                    config.max_loops, 
                                    base=config.loop_rope_base
                )
        self.enc_blocks = nn.ModuleList([EncoderBlock(config, rope=rope) for _ in range(self.n_layer)])
        self.rms_out = RMSNorm(config.n_dim)

    def forward(self, x: List[torch.Tensor]):

        ## TODO: Need to make this incremental
        B, T, C = x[0].size()
        x_reshape = [i.reshape(B*T, C) for i in x]
        x_cat = torch.stack(x_reshape, dim=0).permute(1, 0, 2)


        for idx, block in enumerate(self.enc_blocks):
            # attn_mask is None defaulting to causal mask
            with debug_context(f"Block {idx}", enabled=True):
                x_cat, _ = block(x_cat, attn_mask=None, return_kv_cache=False)

        # Reshape back to (B, T, C)
        output = x_cat[:, -1, :].reshape(B, T, C)
        output = self.rms_out(output)
        return output
    


    def forwardx(self, x: List[torch.Tensor], kv_caches: Optional[List[Tuple[Tensor, Tensor]]]=None):

        ## TODO: Need to make this incremental
        # B, T, C = x[0].size()
        # x_reshape = [i.reshape(B*T, C) for i in x]
        # x_cat = torch.stack(x_reshape, dim=0).permute(1, 0, 2)

        B, T, C = x[-1].size()
        x_cat = x[-1].reshape(B*T, 1, C)
        updated_kv_caches: List[Tuple[Tensor, Tensor]] = []

        if kv_caches is None:
            kv_caches = [None] * self.n_layer

        for idx, block in enumerate(self.enc_blocks):
            # attn_mask is None defaulting to causal mask
            with debug_context(f"Block {idx}", enabled=True):
                x_cat, kv_cache = block(x_cat, attn_mask=None, kv_cache=kv_caches[idx], return_kv_cache=True)

            # if kv_cache is not None:
            updated_kv_caches.append(kv_cache)

        # Reshape back to (B, T, C)

        output = x_cat[:, -1, :].reshape(B, T, C)
        output = self.rms_out(output)
        return output, updated_kv_caches



class REPL(nn.Module):
    def __init__(self,
                config: REPLConfig,
                pad_idx: int = GridTokenizer().PAD_IDX):
        super().__init__()
        self.config = config
        self.n_dim = config.n_dim
        self.max_seq_len = config.max_seq_len
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

        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.inp_collate = IterativeEncoder(config, n_layer=config.n_lecn_layer)
        self.out_collate = IterativeEncoder(config, n_layer=config.n_ldec_layer)

  

        self.lm_head = nn.Linear(config.n_dim, config.grid_vocab_size, bias=False)
        
        self._untrained_block_ids = []

        self.loss = MultiLevelLoss(pad_odx=self.PAD_IDX,
                                edr=config.edr,
                                min_pct=config.mctp)
        
        # weight sharing scheme. Transformer++ (Llama architecture does not tie weights)
        # Reference: https://youtu.be/pRM_P6UfdIc?t=1500
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


    @staticmethod
    def enc_mask(inp_grid: torch.Tensor, program_prefix=3, pad_idx=13):
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
    def dec_enc_mask(inp_grid: torch.Tensor, output_grid, program_prefix=3, pad_idx=13):
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

        program = self.pte(program)
        color_permutation = self.cte(color_permutation)
        array_transform = self.ate(array_transform)
        inp_grid = self.gte(input_grid)

        enc_inp = torch.cat([program, color_permutation, array_transform, inp_grid], dim=1)
        dec_inp = self.gte(output_grid)

        loop_enc_inps = []
        loop_dec_inps = []
        loop_logits = []
        for l in range(n_loops):
            with debug_context(f"Loop {l}", enabled=False):
                enc_inp, enc_kv_cache = self.encoder(enc_inp, enc_mask=enc_mask)
                loop_enc_inps.append(enc_inp)
            with debug_context(f"Loop {l}", enabled=l==1):
                enc_inp = self.inp_collate(loop_enc_inps)

            with debug_context(f"Loop {l}", enabled=False):
                dec_inp, dec_kv_cache = self.decoder(enc_inp, dec_inp, enc_mask=dec_enc_mask, dec_mask=dec_mask)
                loop_dec_inps.append(dec_inp)
                dec_inp = self.out_collate(loop_dec_inps)


            logits = self.lm_head(dec_inp)
            loop_logits.append(logits)

        return loop_logits
    

    def forwardx(self, x: torch.Tensor, n_loops: int = 1):
        color_permutation = x.color_permutation
        array_transform = x.array_transform
        program = x.program
        input_grid = x.input
        output_grid = x.causal_output
        enc_mask = self.enc_mask(input_grid, program_prefix=3, pad_idx=self.PAD_IDX)
        dec_mask = self.dec_mask(output_grid)
        dec_enc_mask = self.dec_enc_mask(input_grid, output_grid, program_prefix=3, pad_idx=self.PAD_IDX)

        program = self.pte(program)
        color_permutation = self.cte(color_permutation)
        array_transform = self.ate(array_transform)
        inp_grid = self.gte(input_grid)

        enc_inp = torch.cat([program, color_permutation, array_transform, inp_grid], dim=1)
        dec_inp = self.gte(output_grid)

        loop_enc_inps = []
        loop_dec_inps = []
        loop_logits = []

        inp_kv_caches = None
        out_kv_caches = None

        for l in range(n_loops):
            with debug_context(f"Loop {l}", enabled=False):
                enc_inp, enc_kv_cache = self.encoder(enc_inp, enc_mask=enc_mask)
                loop_enc_inps.append(enc_inp)
            with debug_context(f"Inc Loop {l}", enabled=l==1 and False):
                enc_inp, inp_kv_caches = self.inp_collate.forwardx(loop_enc_inps, kv_caches=inp_kv_caches)

            with debug_context(f"Loop {l}", enabled=False):
                dec_inp, dec_kv_cache = self.decoder(enc_inp, dec_inp, enc_mask=dec_enc_mask, dec_mask=dec_mask)
                loop_dec_inps.append(dec_inp)
                dec_inp, out_kv_caches = self.out_collate.forwardx(loop_dec_inps, kv_caches=out_kv_caches)
                
            logits = self.lm_head(dec_inp)
            loop_logits.append(logits)

        return loop_logits
    
    def compute_loss(self, loop_logits: List[torch.Tensor], y: torch.Tensor):
        return self.loss(loop_logits, y)

#%%

config = REPLConfig(
    prog_vocab_size=len(arc_tokenizer.program_tokenizer),
    n_dim=64,
    n_head=4,
    n_embd=8,
    n_enc_layer=1,
    n_dec_layer=1,
    n_lecn_layer=1,
    n_ldec_layer=1,
    max_seq_len=2048,
    pnorm=1.0,
    dropout=0.0
)

interpreter = REPL(config)
num_loops = 2

#%%
loop_logits = interpreter(x, num_loops)
loop_logits2 = interpreter.forwardx(x, num_loops)

# loss = interpreter.compute_loss(loop_logits, y.output)
# loss
# %%
# loss2 = interpreter.compute_loss(loop_logits2, y.output)
# loss2
# %%
loss = interpreter.compute_loss(loop_logits, y.output)


#%%

for i in range(len(loop_logits)):
    print(loop_logits[i].shape, loop_logits2[i].shape)
    print((loop_logits[i] - loop_logits2[i]).abs().max())
# %%
loop_logits2
# %%
x.input.shape
# %%
x.program.shape
# %%

def create_test_inp(config, bs=10, inp_seq_len=10, out_seq_len=5, pad_idx=13):

    program = torch.randint(0, config.prog_vocab_size, (bs, 1))
    color_permutation = torch.randint(0, config.perm_vocab_size, (bs, 1))
    array_transform = torch.randint(0, config.tform_vocab_size, (bs, 1))
    input_grid = torch.randint(0, pad_idx-1, (bs, inp_seq_len))

    for b in range(bs):
        input_grid[b, np.random.randint(inp_seq_len//2, inp_seq_len):] = pad_idx

    output_grid = torch.randint(0, pad_idx-1, (bs, out_seq_len))
    output_grid[:, 0] = pad_idx

    return MODEL_INPUT(program=program, color_permutation=color_permutation, array_transform=array_transform, input=input_grid,causal_output=output_grid, meta=None)


x = create_test_inp(config, 1)

x.input.shape
# %%
enc_mask = interpreter.enc_mask(x.input, program_prefix=0, pad_idx=13)
enc_mask.shape
# %%
enc_mask
# %%
x.input
# %%
x.input
# %%
x.causal_output
# %%
torch.ones(1, 2, 1, dtype=torch.bool).tril(diagonal=0)

# %%
