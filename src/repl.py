#%%
from dataclasses import dataclass
import re
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .lazy_adamw import LazyAdamW
from .rope2d import RoPE2D
from .tokenizer import MODEL_INPUT, MODEL_OUTPUT, ArrayTransformTokenizer, ColorPermutationTokenizer, GridTokenizer
from .mask_utils import create_enc_dec_mask
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
    pad_idx: int = GridTokenizer().PAD_IDX
    max_grid_height: int = 60
    max_grid_width: int = 60
    rope_base: int = 10_000 # Base for geometric progression in angle computation
    lora_r: int = 0 # Rank for LoRA linear layer
    lora_alpha: Optional[int] = None # Alpha for LoRA linear layer, it None, it defaults to lora_r
    n_iter: int = 4

    def __post_init__(self):
        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        if self.pnorm is not None:
            assert 0.0 < self.pnorm, "p-norm must be greater than 0"
        
        head_dim = self.n_dim // self.n_head
        assert head_dim % 2 == 0, "Head dimension must be even"

        if self.lora_alpha is None:
            self.lora_alpha = self.lora_r

        assert isinstance(self.lora_r, int) and self.lora_r >= 0, "LoRA rank must be a non-negative integer"


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
            'pad_idx': self.pad_idx,
            'max_grid_height': self.max_grid_height,
            'max_grid_width': self.max_grid_width,
            'rope_base': self.rope_base,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'n_iter': self.n_iter,
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
    def __init__(self, config: REPLConfig, rope: Optional[RoPE2D]=None):
        super().__init__()
        self.config = config
        self.rope = rope
        assert config.n_dim % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_dim, 3 * config.n_dim, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_dim, config.n_dim, bias=False)

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
            k = self.rope(k, positions.unsqueeze(1))
            q = self.rope(q, positions.unsqueeze(1))


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
    def __init__(self, config: REPLConfig, rope: Optional[RoPE2D]):
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


class Interpreter(nn.Module):
    def __init__(self, config: REPLConfig) -> None:
        super().__init__()
        self.config = config

        rope_2d = RoPE2D(config.n_dim // config.n_head,
                        max_height=config.max_grid_height,
                        max_width=config.max_grid_width,
                        base=config.rope_base)
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
            dropout=config.dropout
        )
        self.rnn.flatten_parameters()
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

        # self.rnn.flatten_parameters()
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


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, r=0, alpha=1):
        """
        LoRA-enhanced Linear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
            r (int, optional): Rank of the LoRA decomposition. If 0, LoRA is not applied. Default: 0.
            alpha (int, optional): Scaling factor for LoRA. Default: 1.
        """
        super(LoRALinear, self).__init__(in_features, out_features, bias)
        self.r = r
        self.alpha = alpha

        if self.r > 0:
            # Initialize LoRA parameters A and B
            self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)  # Small random initialization
            self.B = nn.Parameter(torch.zeros(out_features, r))        # Zero initialization
            self.scaling = self.alpha / self.r
        else:
            # Not registering parameter to deliberably cause different state dicts
            self.A = None
            self.B = None
            self.scaling = 0.0


    def fine_tune_mode(self):
        # Freeze the original weights and bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Unfreeze the LoRA parameters
        assert self.r > 0
        self.A.requires_grad = True
        self.B.requires_grad = True

    def forward(self, input):
        """
        Forward pass through the LoRA-enhanced Linear layer.

        Args:
            input (Tensor): Input tensor of shape (*, in_features).

        Returns:
            Tensor: Output tensor of shape (*, out_features).
        """
        if self.A is not None and self.B is not None:
            # Compute the low-rank adaptation
            # W' = W + (B @ A) * scaling
            lora_weight = torch.matmul(self.B, self.A) * self.scaling
            # Adjust the original weight with LoRA weight
            adjusted_weight = self.weight + lora_weight
            return F.linear(input, adjusted_weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, r={self.r}, alpha={self.alpha}')



class REPL(nn.Module):
    def __init__(self,
                config: REPLConfig):
        super().__init__()
        self.config = config
        self.n_iter = config.n_iter
        self.n_dim = config.n_dim
        self.n_layer = config.n_layer
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

        self.lm_head = LoRALinear(config.n_dim, config.grid_vocab_size, bias=False, r=config.lora_r, alpha=config.lora_alpha)
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


    @torch.jit.export
    def get_pte_weight(self):
        return self.pte[0].weight

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
        )-> Tuple[Tensor, Optional[Tuple[List[List[Tuple[Tensor, Tensor]]], Tensor, Tensor]]]:

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

        # iter_outs = []
        updated_kv_cache: List[List[Tuple[Tensor, Tensor]]] = [] 

        current_state = enc_dec_inp

        agg_out = current_state  # Dummy initialisation to make TorchScript happy
        for i in range(self.n_iter):
            interpreter_out, iter_kv_cache = self.interpreter(
                                            x=current_state,
                                            attn_mask=attn_mask,
                                            positions=enc_dec_indices,
                                            kv_cache=None,
                                            return_kv_caches=return_cache)
            
            agg_out, current_state = self.state_agg(interpreter_out, current_state)
            # iter_outs.append(agg_out)
            if iter_kv_cache is not None:
                updated_kv_cache.append(iter_kv_cache)

        # iter_outs = torch.stack(iter_outs, dim=0)
        # logits = self.lm_head(iter_outs[:, :, dec_start_idx:, :])
        logits = self.lm_head(agg_out[:, dec_start_idx:, :])

        cache = (updated_kv_cache, enc_valid_mask, dec_valid_mask) if return_cache else None
        return logits, cache


    @torch.jit.export
    def forward_inc(self,
            next_y: MODEL_OUTPUT, 
            cache: Tuple[List[List[Tuple[Tensor, Tensor]]], Tensor, Tensor],
        ) -> Tuple[Tensor, Tuple[List[List[Tuple[Tensor, Tensor]]], Tensor, Tensor]]:
                   
        dec_inp, dec_valid_mask, dec_indices = self.contruct_decoder_input(next_y)
        seq_len = dec_inp.size(1)
        past_kv_cache, past_enc_valid_mask, past_dec_valid_mask = cache

        dec_valid_mask = torch.cat([past_dec_valid_mask, dec_valid_mask], dim=1)

        attn_mask = create_enc_dec_mask(past_enc_valid_mask, dec_valid_mask).unsqueeze(1)

        # Need to strip the attn_mask to match the size of decoder input
        attn_mask = attn_mask[:, :, -seq_len:, :]

        iter_outs = []
        updated_kv_cache: List[List[Tuple[Tensor, Tensor]]] = []

        current_state = dec_inp
        agg_out = current_state  # Dummy initialisation to make TorchScript happy
        for i in range(self.n_iter):
            interpreter_out, iter_kv_cache = self.interpreter(
                                            x=current_state,
                                            attn_mask=attn_mask,
                                            positions=dec_indices,
                                            kv_cache=past_kv_cache[i],
                                            return_kv_caches=True)
            
            agg_out, current_state = self.state_agg(interpreter_out, current_state)
            iter_outs.append(agg_out)
            # Store the updated kv-cache for this loop iteration
            updated_kv_cache.append(iter_kv_cache)

        # iter_outs = torch.stack(iter_outs, dim=0)
        # logits = self.lm_head(iter_outs)
        logits = self.lm_head(agg_out)

        cache = (updated_kv_cache, past_enc_valid_mask, dec_valid_mask)
        return logits, cache

    
    @torch.jit.export
    def greedy_search(self, 
            input_grid: List[int],
            input_indices: List[List[int]],
            prog_idx: int = 0,
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
        # input_indices_list = [list(t) for t in input_indices]

        x = MODEL_INPUT(
            program=torch.tensor([[prog_idx]], dtype=torch.long, device=device),
            color_permutation=torch.tensor([[color_perm_idx]], dtype=torch.long, device=device),
            array_transform=torch.tensor([[array_tform_idx]], dtype=torch.long, device=device),
            grid=torch.tensor([input_grid], dtype=torch.long, device=device),
            grid_indices=torch.tensor([input_indices], dtype=torch.long, device=device),  # Fixed line
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
            # logits = logits_iters[-1]  
            logits = logits_iters

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

        prefix_list: List[int] = [bos_idx]
        output_list: List[int] = output_sequence.tolist()  # Use .tolist() now since it's supported in TorchScript
        output_list = prefix_list + output_list
        torch.set_grad_enabled(True)
        return output_list, torch.exp(output_log_prob)
    
    @staticmethod
    def _select_kv_caches(kv_caches: List[List[Tuple[torch.Tensor, torch.Tensor]]], mask_or_indices: torch.Tensor) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        # Selects the kv_caches for a specific beam index
        selected_kv_caches: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []
        for loop_kv in kv_caches:
            selected_loop_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for k, v in loop_kv:
                if isinstance(mask_or_indices, torch.Tensor) and mask_or_indices.dtype == torch.bool:
                    k = k[mask_or_indices]
                    v = v[mask_or_indices]
                elif isinstance(mask_or_indices, torch.Tensor) and mask_or_indices.dtype == torch.long:
                    k = torch.index_select(k, 0, mask_or_indices)
                    v = torch.index_select(v, 0, mask_or_indices)
                    assert v.size(0) == mask_or_indices.size(0), "Gathering didn't work!"
                else:
                    raise ValueError(f"mask_or_indices must be a tensor of type bool or long. Got {mask_or_indices} ")
                selected_loop_kv.append((k, v))
            selected_kv_caches.append(selected_loop_kv)
        return selected_kv_caches
    
    @torch.jit.export
    def beam_search(self, 
        input_grid: List[int],
        input_indices: List[List[int]],
        top_k: int = 5,
        prob_thresh: float = 0.001,  # Finish when the probability of the top beam is below this threshold
        prog_idx: int = 0,
        color_perm_idx: int = 0,
        array_tform_idx: int = 0,
        bos_idx: int = GridTokenizer().BOS_IDX,
        eos_idx: int = GridTokenizer().EOS_IDX,
        new_row_idx: int = GridTokenizer().NEW_ROW_IDX,
        max_grid_height: int = 35,
        max_grid_width: int = 35)-> Tuple[List[List[int]], List[float]]:
    
        # Compute log_prob_threshold from prob_threshold
        log_prob_thresh = -float('inf') if prob_thresh == 0.0 else torch.log(prob_thresh) 

        # Assume prog_idx and inp_idx are lists of integers
        # Batch size is 1
        device = self.type_emb.weight.device  # Get the device from the embedding layer (assuming it's available)

        # device = self.ate[0].weight.device  # Get the device from the embedding layer (assuming it's available)
        max_length = max_grid_height * max_grid_width
        max_r = torch.tensor(max_grid_height-1, device=device)
        max_c = torch.tensor(max_grid_width-1, device=device)
        torch.set_grad_enabled(False)

        # Convert input_indices to a list of lists to make torchscript happy
        # input_indices_list = [list(t) for t in input_indices]

        x = MODEL_INPUT(
            program=torch.tensor([[prog_idx]], dtype=torch.long, device=device),
            color_permutation=torch.tensor([[color_perm_idx]], dtype=torch.long, device=device),
            array_transform=torch.tensor([[array_tform_idx]], dtype=torch.long, device=device),
            grid=torch.tensor([input_grid], dtype=torch.long, device=device),
            grid_indices=torch.tensor([input_indices], dtype=torch.long, device=device),  # Fixed line
            meta=None
        )

        _, cache = self.forward(
                x=x,
                y=None,
                return_cache=True
        )

        assert cache is not None, "Cache must be returned for greedy search"

        last_token_indices = torch.tensor([[[0, 0]]], dtype=torch.long, device=device)
        last_token = torch.tensor([[bos_idx]], dtype=torch.long, device=device)

        next_y = MODEL_OUTPUT(
            grid=last_token,  # Shape: (1, 1),
            grid_indices=last_token_indices,  # Shape: (1, 1, 2)
            target_grid=None
        )

        output_sequence = torch.zeros(1, 0, dtype=torch.long, device=device)  # Shape: (1,)
        output_log_probs = torch.zeros(1, 0, dtype=torch.float, device=device)  # Shape: (1,)

        candidate_sequences: List[List[int]] = []  # Separate list for sequences
        candidate_log_probs: List[float] = []  # Separate list for log probabilities

        for t in range(max_length):
            # # if torch.cuda.is_available():
            # torch.cuda.synchronize() # wait for the GPU to finish work
            # torch.cuda.empty_cache()

            next_y = MODEL_OUTPUT(
                grid=last_token,  # Shape: (1, 1),
                grid_indices=last_token_indices,  # Shape: (1, 1, 2)
                target_grid=None
            )

            logits_iters, cache = self.forward_inc(
                next_y=next_y,
                cache=cache
            )
            # logits = logits_iters[-1]
            logits = logits_iters
            seq_len = output_sequence.size(1)
            bs = output_sequence.size(0)

            # Get the logits for the last token
            next_logits = logits[:, -1, :]  # Shape: (1, vocab_size)
            log_probs = F.log_softmax(next_logits, dim=-1)  # Shape: (1, vocab_size)

            # Convert log probabilities to probabilities for sampling
            probs = torch.exp(log_probs)

            sample_k_tokens = torch.multinomial(probs, top_k, replacement=False)
            # Gather the log probabilities of the sampled tokens for each sequence
            batch_indices = torch.arange(log_probs.size(0)).unsqueeze(-1).expand(-1, top_k)
            sample_log_probs = log_probs[batch_indices, sample_k_tokens]

            # Get top top_k tokens and their log probabilities
            # topk_log_probs, topk_tokens = log_probs.topk(top_k, dim=-1)  # Each is (1, beam_width)

            topk_log_probs, topk_tokens = sample_log_probs, sample_k_tokens

            # Expand the sequences and log_probs
            output_sequence = output_sequence.unsqueeze(1).expand(bs, top_k, seq_len)  # Shape: (bs,  top_k, seq_len)
            output_log_probs = output_log_probs.unsqueeze(1).expand(bs, top_k, seq_len)  # Shape: (bs, top_k, seq_len)


            # Combine the expanded sequences with the new tokens
            output_sequence = torch.cat([output_sequence, topk_tokens.unsqueeze(-1)], dim=-1)  # Shape: (top_k, seq_len+1, top_k)
            output_log_probs = torch.cat([output_log_probs, topk_log_probs.unsqueeze(-1)], dim=-1)  # Shape: (top_k, seq_len+1, top_k)

            # reshape the output sequence and log_probs
            output_sequence = output_sequence.reshape(bs * top_k, -1)  # Shape: (top_k*bs, seq_len+1)
            output_log_probs = output_log_probs.reshape(bs * top_k, -1)  # Shape: (top_k*bs, seq_len+1)

            sorted_log_probs, sorted_indices = output_log_probs.sum(dim=-1).sort(descending=True)  # Shape: (top_k^2,)

            # Select the top top_k sequences
            topk_indices = sorted_indices[:top_k]  # Shape: (top_k,)
            original_indices = topk_indices // top_k  # Shape: (top_k,)

            # Update the output sequence and log_probs
            output_sequence = output_sequence[topk_indices]
            output_log_probs = output_log_probs[topk_indices]

            mask_ends_eos = output_sequence[:, -1] == eos_idx

            # Separate the sequences that end with EOS token
            completed_sequences = output_sequence[mask_ends_eos]
            completed_log_probs = output_log_probs[mask_ends_eos].sum(dim=-1)


            for seq, log_prob in zip(completed_sequences, completed_log_probs):
                seq_list: List[int] = seq.tolist()
                log_prob: float = float(log_prob.item())
                candidate_sequences.append(seq_list)
                candidate_log_probs.append(log_prob)

            # Keep the sequences that do not end with EOS token
            output_sequence = output_sequence[~mask_ends_eos]
            output_log_probs = output_log_probs[~mask_ends_eos]

            # If all sequences are completed, stop
            if output_sequence.size(0) == 0:
                break

            if torch.all(output_log_probs.sum(dim=-1) < log_prob_thresh):
                break

            # If not, then prepare the next input
            kv_caches, enc_mask, dec_mask = cache

            # reconstruct kv_caches based on the original indices
            kv_caches = self._select_kv_caches(kv_caches, original_indices)
            enc_mask = torch.index_select(enc_mask, 0, original_indices)
            dec_mask = torch.index_select(dec_mask, 0, original_indices)
            last_token_indices = torch.index_select(last_token_indices, 0, original_indices)

            # The apply the non-eos mask
            kv_caches = self._select_kv_caches(kv_caches, ~mask_ends_eos)
            enc_mask = enc_mask[~mask_ends_eos]
            dec_mask = dec_mask[~mask_ends_eos]
            last_token_indices = last_token_indices[~mask_ends_eos]

            # Now update the last token and indices
            last_token = output_sequence[:, -1].unsqueeze(1)

            # Create a boolean mask where last_token equals new_row_idx
            mask_new_row = (last_token[:, 0] == new_row_idx)  # Shape: [B]

            # Update row indices: If mask is True, increment by 1 and clamp to max_r else, keep the original row index
            updated_rows = torch.where(
                mask_new_row,
                torch.min(last_token_indices[:, 0, 0] + 1, max_r),
                last_token_indices[:, 0, 0]
            )

            # Update column indices: If mask is True, set to 0 Else, increment by 1 and clamp to max_c
            updated_cols = torch.where(
                mask_new_row,
                torch.tensor(0, device=device),
                torch.min(last_token_indices[:, 0, 1] + 1, max_c)
            )

            # Combine the updated rows and columns
            last_token_indices = torch.stack([updated_rows, updated_cols], dim=1).unsqueeze(1)  # Shape: [B, 1, 2]

            # Repack the cache
            cache = (kv_caches, enc_mask, dec_mask)

        # This particular way is used for TorchScript compatibility
        # Sort the candidate_log_probs and reorder candidate_sequences based on the sorted candidate_log_probs
        sorted_indices: List[int] = torch.tensor(candidate_log_probs).argsort(descending=True).tolist()  # Sort indices based on log_probs
        sorted_probs = [math.exp(candidate_log_probs[i]) for i in sorted_indices]
        sorted_sequences = [[bos_idx] + candidate_sequences[i] for i in sorted_indices]

        # Combine sorted log_probs and sequences into the final output
        # output_candidates = [(seq, log_prob) for log_prob, seq in zip(sorted_log_probs, sorted_sequences)]

        torch.set_grad_enabled(True)
        return sorted_sequences, sorted_probs
    
    def get_optimizer(
            self, 
            model_lr,
            prog_lr,
            model_wd,
            prog_wd=0.0,
            prog_l1=0.0,
            device_type=None
        ):

        if model_lr == 0:
            assert prog_lr > 0, "Program learning rate must be greater than 0"
            self.fine_tune_mode(enable_lora=True if self.config.lora_r > 0 else False)

        program_param_keys = ['pte.0.weight', 'lm_head.A', 'lm_head.B']

        # Separate the embedding parameters
        program_params = [p for n, p in self.named_parameters() if n in program_param_keys and p.requires_grad]
        model_params = [p for n, p in self.named_parameters() if n not in program_param_keys and p.requires_grad]

        if self.config.lora_r > 0:
            assert len(program_params) == 3, "Program parameters must be 3"
        else:
            assert len(program_params) == 1, "Program parameters must be 1"

        if model_lr == 0:
            assert len(model_params) == 0, "When model_lr == 0, there shouldn't be any model params to train!"

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

    def fine_tune_mode(self, enable_lora: bool = False):
        # Freeze model params if model_lr is 0, needed for finetuning
        logger.warning("Freezing model parameters. Only Program embedding parameters will be trained.")
        logger.warning("This setting should only be used for training without resuming/forkin (with optimizer state load disabled)")

        for n, p in self.named_parameters():
            if 'pte.0' not in n:
                p.requires_grad = False

        if enable_lora:
            logger.warning("Unfreezing LoRA parameters for fine-tuning.")
            self.lm_head.fine_tune_mode()

    def load_state_dict(self, state_dict, strict: bool=True, assign: bool=False):

        if strict:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        
        def skip_if_not_present(key):
            if key not in state_dict and key in self.state_dict():
                logger.warning(f"WARNING: Skipping loading {key} as it is not present in the source model.")
                state_dict[key] = self.state_dict()[key]

        skip_if_not_present('lm_head.A')
        skip_if_not_present('lm_head.B')

        pte_w_key = 'pte.0.weight'
        if self.state_dict()[pte_w_key].shape != state_dict[pte_w_key].shape:
            logger.warning(f"WARNING: Skipping loading Program Embeddings as they differ in shape in the target model. ")
            state_dict = {k: v for k, v in state_dict.items() if k != pte_w_key}
            state_dict[pte_w_key] = self.state_dict()[pte_w_key]


        def get_num_blocks(sd, prefix):
            max_block_id = 0

            for key in sd.keys():
                # Use regex to match the pattern
                match = re.match(rf'{prefix}\.blocks\.(\d+)\.', key)
                if match:
                    block_number = int(match.group(1))
                    max_block_id = max(max_block_id, block_number)

            return max_block_id + 1

        @torch.no_grad()
        def copy_(prefix, idx_mapping=None, src_prefix=None):
            for name, t in self.state_dict().items():
                if name.startswith(prefix):
                    suffix = name[len(prefix):]
                    src_name = src_prefix + suffix if src_prefix is not None else name
                    s = state_dict[src_name]
                    trg_ptr_b4 = t.data_ptr()
                    if idx_mapping is None:
                        t.data.copy_(s)
                    else:
                        for trg_idx, src_idx in idx_mapping.items():
                            t.data[trg_idx].copy_(s.data[src_idx])

                    trg_ptr_after = t.data_ptr()
                    assert trg_ptr_b4 == trg_ptr_after, f"Data pointer changed for {prefix}"

        missing_keys, unexpected_keys = super().load_state_dict(state_dict, strict=strict, assign=assign)

        for k in missing_keys + unexpected_keys:
            assert k.startswith('interpreter.') or k.startswith('state_agg.'), f"Unexpected/Missing key: {k}"

        def deal_with_blocks(prefix):
            num_trg_blocks = get_num_blocks(self.state_dict(), prefix)
            num_src_blocks = get_num_blocks(state_dict, prefix)
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