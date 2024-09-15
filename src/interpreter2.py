#%%
from dataclasses import dataclass
import inspect
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.types
from .utils import is_power_of_two, get_logger, gather_along_zero_dim
from .dataset import ProgramTokenizer, GridTokenizer
from torch import Tensor
from torch.cuda.amp import autocast


logger = get_logger()

@dataclass
class InterpreterConfig:
    prog_vocab_size: int # number of program tokens
    grid_vocab_size: int  # number of array element tokens (one extra for niceness)
    n_dim: int  # dimension of the model
    n_head: int # number of heads within each self-attention block
    n_layer: int = 1 # number of transformer blocks / layers
    max_seq_len: int = 2048 # max sequence length
    dropout: float = 0.0 # dropout probability

    def __post_init__(self):
        assert is_power_of_two(self.prog_vocab_size), "Program vocab size must be a power of 2"
        assert is_power_of_two(self.grid_vocab_size), "Grid vocab size must be a power of 2"
        # assert is_power_of_two(self.n_dim), "Model dimension must be a power of 2"

        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")
        
        head_dim = self.n_dim // self.n_head
        assert head_dim % 2 == 0, "Head dimension must be even"


    def to_dict(self):
        return {
            'prog_vocab_size': self.prog_vocab_size,
            'grid_vocab_size': self.grid_vocab_size,
            'n_dim': self.n_dim,
            'n_head': self.n_head,
            'n_layer': self.n_layer,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout
        }
    
    @staticmethod
    def from_dict(data: dict) -> "InterpreterConfig":
        return InterpreterConfig(**data)



class RotaryPositionalEmbeddings(nn.Module):
    """
    Source: https://pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html#RotaryPositionalEmbeddings

    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ````embed_dim`` // ``num_heads````
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    # We need to explicitly define reset_parameters for FSDP initialization, see
    # https://github.com/pytorch/pytorch/blob/797d4fbdf423dd9320ebe383fb57ffb1135c4a99/torch/distributed/fsdp/_init_utils.py#L885
    def reset_parameters(self):
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    @autocast(enabled = False)
    def forward(self, x: Tensor, *, input_pos: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, n_h, h_d]
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with RoPE applied

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim

        TODO: The implementation below can be made more efficient
        for inference.
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, config: InterpreterConfig, rope: RotaryPositionalEmbeddings):
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


    def forward(self, x, attn_mask=None, kv_cache=None, return_kv_cache=False):
        # x: (B, T, C)
        B, T, C = x.size()

        # qkv: (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_dim, dim=2)

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, T, self.n_head, C // self.n_head)  # (B, T, n_head, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)

        # If kv_cache is present, concatenate past keys and values BEFORE applying RoPE
        if kv_cache is not None:
            past_k, past_v = kv_cache  # K: (B, T_past, n_head, head_dim), V: (B, n_head, T_past, head_dim)
            k = torch.cat([past_k, k], dim=1)  # Concatenate along sequence length dimension
            v = torch.cat([past_v, v], dim=2)

        # Update new_kv_cache
        new_kv_cache = (k, v) if return_kv_cache else None

        # Generate position indices for the concatenated sequence
        total_seq_len = k.size(1)  # k now contains both past and current
        position_ids = torch.arange(total_seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand(B, 1, total_seq_len)

        # Apply RoPE to q and k before transposing for attention
        # For q, we use positions corresponding to the current tokens (last T positions)
        q_positions = position_ids[:, :, -T:]  # Shape: (B, 1, T)
        q = self.rope(q, input_pos=q_positions)

        # For k, we use positions for the entire sequence (past + current)
        k_positions = position_ids  # Shape: (B, 1, total_seq_len)
        k = self.rope(k, input_pos=k_positions)

        # Now transpose q and k for attention computation
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)  # (B, n_head, total_seq_len, head_dim)
        
        # Compute attention
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.config.dropout)

        # attn_output: (B, n_head, T, head_dim)
        # Reshape back to (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(attn_output)

        return y, new_kv_cache



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



class TransformerBlock(nn.Module):
    def __init__(self, config: InterpreterConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.rmsnorm = RMSNorm(config.n_dim)
        self.attn = SelfAttention(config, rope)
        self.normed_mlp = nn.Sequential(
                            RMSNorm(config.n_dim),
                            SwiGLUFFN(config.n_dim, 4 * config.n_dim))

    def forward(self, x, attn_mask=None, kv_cache=None, return_kv_cache=False):
        attn_output, new_kv_cache = self.attn(self.rmsnorm(x), attn_mask=attn_mask, kv_cache=kv_cache, return_kv_cache=return_kv_cache)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.normed_mlp(x))
        return x, new_kv_cache


class Interpreter(nn.Module):
    def __init__(self,
                config: InterpreterConfig,
                prog_tokenizer: ProgramTokenizer = None,
                grid_tokenizer: GridTokenizer = None):
        super().__init__()
        self.config = config
        self.prog_tokenizer = prog_tokenizer
        self.grid_tokenizer = grid_tokenizer

        self.pte = nn.Embedding(config.prog_vocab_size, config.n_dim)
        self.wte = nn.Embedding(config.grid_vocab_size, config.n_dim)
        
        rope = RotaryPositionalEmbeddings(config.n_dim // config.n_head, config.max_seq_len)
        
        self.inp_inject = nn.Linear(2*config.n_dim, config.n_dim, bias=False)

        self.blocks = nn.ModuleList([TransformerBlock(config, rope=rope) for _ in range(config.n_layer)])
        self.ln_f = RMSNorm(config.n_dim)

        self.lm_head = nn.Linear(config.n_dim, config.grid_vocab_size, bias=False)
        
        
        # weight sharing scheme. Transformer++ (Llama architecture does not tie weights)
        # Reference: https://youtu.be/pRM_P6UfdIc?t=1500
        # self.wte.weight = self.lm_head.weight

        # init params
        # self.apply(self._init_weights)

    @staticmethod
    def get_attn_mask(batch_size, src_len, trg_len, device, non_causal_prefix_len):
        # CREATE ATTENTION MASK Shape [bs, trg_len, src_len]
        offset = src_len - trg_len
        causal_mask = torch.ones(batch_size, trg_len, src_len, dtype=torch.bool, device=device).tril(diagonal=offset)

        if non_causal_prefix_len is None:
            return causal_mask.unsqueeze(1).to(device)

        # Create a range tensor for comparison
        idx = torch.arange(src_len, device=device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, src_len]

        # Expand inp_lens to match the dimensions for broadcasting
        expanded_inp_lens = non_causal_prefix_len.unsqueeze(1).unsqueeze(2)  # Shape: [bs, 1, 1]

        # Create the full mask by using broadcasting
        non_causal_mask = (idx < expanded_inp_lens).expand(-1, trg_len, -1)  # Shape: [bs, trg_len, src_len]

        # # Combine the lower triangular mask with the full mask
        attn_mask = causal_mask | non_causal_mask

        return attn_mask.unsqueeze(1).to(device)  # Shape: [bs, 1, trg_len, src_len]


    def forward(self, prog_idx, inp_idx, inp_len, n_loops, max_grad_loops=None, return_convergence_mse=False, return_kv_caches=False):
        # idx is of shape (B, T)

        B1, T1 = prog_idx.size()
        B2, T2 = inp_idx.size()
        batch_seq_len = T1 + T2
        assert B1 == B2, "Batch size of program and input must match"
        assert T1 == 1, "Program input must be a single token"

        assert batch_seq_len <= self.config.max_seq_len, f"Cannot forward sequence of length {batch_seq_len}, max_seq_len is only {self.config.max_seq_len}"

        assert n_loops > 0, "Number of loops must be greater than 0"
        if max_grad_loops is None or max_grad_loops > n_loops: 
            max_grad_loops = n_loops

        assert 0 < max_grad_loops <= n_loops, "max_grad_loops must be less than or equal to n_loops and greater than 0"  

        grad_loop_start = n_loops - max_grad_loops


        # forward the token and position embeddings
        prog_emb = self.pte(prog_idx)  # (B, T1, n_dim)
        inp_emb = self.wte(inp_idx)    # (B, T2, n_dim)
        x = torch.cat((prog_emb, inp_emb), dim=1)

        attn_mask = self.get_attn_mask(
                        batch_size=x.size(0),
                        src_len=batch_seq_len,
                        trg_len=x.size(1),
                        device=x.device,
                        non_causal_prefix_len=inp_len)

        output = torch.zeros_like(x)
        convergence_mse = []
        updated_kv_caches = []

        for loop_id in range(n_loops):
            prev_output = output

            loop_kv_caches = []

            with torch.set_grad_enabled(loop_id >= grad_loop_start and self.training):
                # Inject input and output from previous loop iteration
                layer_inp = torch.cat((x, output), dim=-1)  # (B, T, 2 * n_dim)
                output = self.inp_inject(layer_inp)         # (B, T, n_dim)

                for i, block in enumerate(self.blocks):
                    # Pass the kv_caches from the specific loop_id
                    output, new_kv_cache = block(output, attn_mask=attn_mask, return_kv_cache=return_kv_caches)

                    # Store the updated kv-cache for the current block in the current loop
                    loop_kv_caches.append(new_kv_cache)

            # Store the updated kv-cache for this loop iteration
            updated_kv_caches.append(loop_kv_caches)

            if return_convergence_mse:
                # Update MSE to track convergence between loops
                output_mse = F.mse_loss(output, prev_output)
                convergence_mse.append(output_mse.item())

        # Forward the final layernorm and the classifier
        output = self.ln_f(output)
        logits = self.lm_head(output)  # (B, T, vocab_size)

        final_output = [logits]

        if return_convergence_mse:
            final_output.append(convergence_mse)

        if return_kv_caches:
            final_output.append(updated_kv_caches)

        return tuple(final_output) if len(final_output) > 1 else final_output[0]


    def forward_inc(self, next_input_idx, kv_caches, n_loops, non_causal_prefix_len=None):
        assert len(kv_caches) == n_loops, "Number of kv_caches count must match number of loops"
        assert len(kv_caches[0]) == self.config.n_layer, "Number of kv_caches blocks must match number of layers"
        past_key = kv_caches[0][0][0]
        T_past = past_key.size(1)
        T_future = next_input_idx.size(1)
        batch_seq_len = T_past + T_future
        B1 = past_key.size(0)
        B2 = next_input_idx.size(0)
        assert B1 == B2, "Batch size of kv_caches and next_input_idx must match"

        assert batch_seq_len <= self.config.max_seq_len, f"Cannot forward sequence of length {batch_seq_len}, max_seq_len is only {self.config.max_seq_len}"
        assert n_loops > 0, "Number of loops must be greater than 0"

        x = self.wte(next_input_idx)  # (B, T3, n_dim)

        attn_mask = self.get_attn_mask(
                        batch_size=x.size(0),
                        src_len=batch_seq_len,
                        trg_len=x.size(1),
                        device=x.device, 
                        non_causal_prefix_len=non_causal_prefix_len)

        output = torch.zeros_like(x)

        # List to store the updated kv-cache for each loop
        updated_kv_caches = []

        for loop_id in range(n_loops):
            prev_output = output

            # Create a new list of kv_caches for the current loop
            loop_kv_caches = []

            # Inject input and output from previous loop iteration
            layer_inp = torch.cat((x, output), dim=-1)  # (B, T, 2 * n_dim)
            output = self.inp_inject(layer_inp)         # (B, T, n_dim)

            # new_kv_caches = []
            for i, block in enumerate(self.blocks):
                # Pass the kv_caches from the specific loop_id
                output, new_kv_cache = block(output, attn_mask=attn_mask, kv_cache=kv_caches[loop_id][i], return_kv_cache=True)

                # Store the updated kv-cache for the current block in the current loop
                loop_kv_caches.append(new_kv_cache)

            # Store the updated kv-cache for this loop iteration
            updated_kv_caches.append(loop_kv_caches)

        # Forward the final layernorm and the classifier
        output = self.ln_f(output)
        logits = self.lm_head(output)  # (B, T, vocab_size)

        return logits, updated_kv_caches


    @torch.no_grad()
    def greedy_search(self, prog_idx, inp_idx, n_loops, max_length, eos_token_id=12):
        # Assume prog_idx and inp_idx are lists of integers
        # Batch size is 1

        device = next(self.parameters()).device  # Get the device (CPU or GPU) from the model parameters

        # Compute inp_len as len(prog_idx) + len(inp_idx)
        inp_len = torch.tensor([len(prog_idx) + len(inp_idx)], dtype=torch.long, device=device)  # Shape: (1,)

        # Convert prog_idx and inp_idx to tensors and add batch dimension
        prog_idx = torch.tensor([prog_idx], dtype=torch.long, device=device)  # Shape: (1, len(prog_idx))

        # Add EOS token to the end of the input sequence
        inp_idx = torch.tensor([inp_idx], dtype=torch.long, device=device)    # Shape: (1, len(inp_idx))

        logits, kv_caches = self.forward(
                prog_idx,
                inp_idx,
                inp_len,
                n_loops,
                return_kv_caches=True
            )

        # Compute inp_len
        # Initialize the output sequence and kv_caches
        pad_token_id = self.grid_tokenizer.PAD_IDX
        last_token = pad_token_id

        output_sequence = torch.tensor([], dtype=torch.long, device=device)  # Shape: (seq_len,)

        for t in range(max_length):
            # Prepare the next input (the last generated token)
            if output_sequence.size(0) > 0:
                last_token = output_sequence[-1].item()

            next_input_idx = torch.tensor([[last_token]], dtype=torch.long, device=device)  # Shape: (1, 1)

            # Run the model forward
            logits, kv_caches = self.forward_inc(
                next_input_idx=next_input_idx,
                kv_caches=kv_caches,
                n_loops=n_loops
            )

            # Get the logits for the last token
            next_logits = logits[:, -1, :]  # Shape: (1, vocab_size)

            # Select the token with the highest probability
            top_token = torch.argmax(next_logits, dim=-1)  # Shape: (1,)

            # Append the new token to the output sequence
            output_sequence = torch.cat([output_sequence, top_token])

            # If EOS token is generated, stop
            if top_token.item() == eos_token_id:
                break

        return output_sequence.tolist()


    @torch.no_grad()
    def beam_search(self, prog_idx, inp_idx, n_loops, max_length, top_k=5, eos_token_id=12):
        # Assume prog_idx and inp_idx are lists of integers
        # Batch size is 1

        def _select_kv_caches(kv_caches, mask_or_indices):
            # Selects the kv_caches for a specific beam index
            selected_kv_caches = []
            for loop_kv in kv_caches:
                selected_loop_kv = []
                for k, v in loop_kv:
                    if isinstance(mask_or_indices, torch.BoolTensor):
                        k = k[mask_or_indices]
                        v = v[mask_or_indices]
                    elif isinstance(mask_or_indices, torch.LongTensor):
                        k = gather_along_zero_dim(k, mask_or_indices)
                        v = gather_along_zero_dim(v, mask_or_indices)
                    else:
                        raise ValueError("mask_or_indices must be a tensor of type bool or long")
                    selected_loop_kv.append((k, v))
                selected_kv_caches.append(selected_loop_kv)
            return selected_kv_caches

        device = next(self.parameters()).device  # Get the device (CPU or GPU) from the model parameters

        # Compute inp_len as len(prog_idx) + len(inp_idx)
        inp_len = torch.tensor([len(prog_idx) + len(inp_idx)], dtype=torch.long, device=device)  # Shape: (1,)

        # Convert prog_idx and inp_idx to tensors and add batch dimension
        prog_idx = torch.tensor([prog_idx], dtype=torch.long, device=device)  # Shape: (1, len(prog_idx))

        # Add EOS token to the end of the input sequence
        inp_idx = torch.tensor([inp_idx], dtype=torch.long, device=device)    # Shape: (1, len(inp_idx))

        logits, kv_caches = self.forward(
                prog_idx,
                inp_idx,
                inp_len,
                n_loops,
                return_kv_caches=True
            )

        # Compute inp_len
        # Initialize the output sequence and kv_caches
        pad_token_id = self.grid_tokenizer.PAD_IDX
        last_token = pad_token_id

        next_input_idx = torch.tensor([[last_token]], dtype=torch.long, device=device) # Shape: (1, 1)

        output_sequence = torch.zeros(1, 0, dtype=torch.long, device=device)  # Shape: (1,)
        output_log_probs = torch.zeros(1, 0, dtype=torch.float, device=device)  # Shape: (1,)

        output_candidates = []

        for t in range(max_length):
            # Prepare the next input (the last generated token)
            if output_sequence.size(1) > 0:
                # last_token = output_sequence[-1].item()
                next_input_idx = output_sequence[:, -1].unsqueeze(1)
            
            # Run the model forward
            logits, kv_caches = self.forward_inc(
                next_input_idx=next_input_idx,
                kv_caches=kv_caches,
                n_loops=n_loops)

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

            _, sorted_indices = output_log_probs.sum(dim=-1).sort(descending=True)  # Shape: (top_k^2,)

            # Select the top top_k sequences
            topk_indices = sorted_indices[:top_k]  # Shape: (top_k,)

            original_indices = topk_indices // top_k  # Shape: (top_k,)

            # Update the output sequence and log_probs
            output_sequence = output_sequence[topk_indices]
            output_log_probs = output_log_probs[topk_indices]

            # reconstruct kv_caches based on the original indices
            kv_caches = _select_kv_caches(kv_caches, original_indices)
            mask_ends_eos = output_sequence[:, -1] == 12

            # Separate the sequences that end with EOS token
            completed_sequences = output_sequence[mask_ends_eos]
            completed_log_probs = output_log_probs[mask_ends_eos].sum(dim=-1)

            for seq, log_prob in zip(completed_sequences, completed_log_probs):
                output_candidates.append((seq.tolist(), log_prob.item()))

            output_sequence = output_sequence[~mask_ends_eos]
            output_log_probs = output_log_probs[~mask_ends_eos]

            kv_caches = _select_kv_caches(kv_caches, ~mask_ends_eos)
            
            # If all sequences are completed, stop
            if output_sequence.size(0) == 0:
                break

        # Sort the output candidates by their log probabilities
        output_candidates = sorted(output_candidates, key=lambda x: x[1], reverse=True)

        # print("Output candidates:", output_candidates)
        return output_candidates


    def loss_fn(self, logits, targets):
        loss =  F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none').view_as(targets)   # This will be a float tensor
        mask = targets != self.grid_tokenizer.PAD_IDX
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def get_optimizer(
                self, 
                model_lr,
                prog_lr,
                model_wd,
                prog_wd=0.0,
                device_type=None,
            ):

        # Freeze model params if model_lr is 0, needed for finetuning
        if model_lr == 0.0:
            logger.warning("Freezing model parameters. Only embedding parameters will be trained.")
            logger.warning("This setting should only be used for training without resuming/forkin (with optimizer state load disabled)")

            for n, p in self.named_parameters():
                if 'pte' not in n:
                    p.requires_grad = False

        # Separate the embedding parameters
        program_params = [p for n, p in self.named_parameters() if 'pte' in n and p.requires_grad]
        model_params = [p for n, p in self.named_parameters() if 'pte' not in n and p.requires_grad]

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in model_params if p.dim() >= 2]
        nodecay_params = [p for p in model_params if p.dim() < 2]
        optim_groups = [
            {'params': program_params,
             'lr': prog_lr,
              'weight_decay': prog_wd},
            {'params': decay_params,
              'lr': model_lr,
              'weight_decay': model_wd},
            {'params': nodecay_params,
              'lr': model_lr,
              'weight_decay': 0.0}]
        # Create AdamW optimizer and use the fused version if it is available
        use_fused = False
        if torch.cuda.is_available() and (device_type is None or device_type == 'cuda'):
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available
            print(f"Using fused AdamW: {use_fused}")
            
        optimizer = torch.optim.AdamW(optim_groups, lr=model_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


    def check_compatible(self, other: "Interpreter", strict=True) -> bool:
        assert isinstance(other, Interpreter), "Can only compare with another Interpreter instance"
        
        config: InterpreterConfig = self.config
        other_config: InterpreterConfig = other.config

        and_conditions = [
            (config.n_dim == other_config.n_dim, "Model dimension should be the same"),
            (config.n_head == other_config.n_head, "Number of heads should be the same"),
        ]

        for cond, msg in and_conditions:
            assert cond, msg
        
        strict_conditions = [
            (self.prog_tokenizer == other.prog_tokenizer, "Program tokenizers should be the same"),
            (self.grid_tokenizer == other.grid_tokenizer, "Grid tokenizers should be the same"),
            (config.grid_vocab_size == other_config.grid_vocab_size, "Grid vocab size should be the same"),
            (config.prog_vocab_size == other_config.prog_vocab_size, "Program vocab size should be the same"),
            (config.n_layer == other_config.n_layer, "Number of blocks should be the same"),
            (config.max_seq_len == other_config.max_seq_len, "Max sequence length should be the same"),
        ]

        for cond, msg in strict_conditions:
            if strict:
                assert cond, msg
            else:
                if not cond:
                    logger.warning(msg + " but they are not. Continuing anyway due to strict=False")

        return all(and_conditions)
    

    def load_from_model(self, src_model: "Interpreter", strict=True) -> None:
        assert self.check_compatible(src_model, strict=strict), "Models are not compatible"

        # keep_vars=True to keep the requires_grad flag
        config_trg = self.config
        config_src = src_model.config
        trg_sd = self.state_dict(keep_vars=True) 
        src_sd = src_model.state_dict()

        trg_grid_token2idx = self.grid_tokenizer.token2idx
        src_grid_token2idx = src_model.grid_tokenizer.token2idx
        trg_prog_token2idx = self.prog_tokenizer.token2idx
        src_prog_token2idx = src_model.prog_tokenizer.token2idx

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
                            trg_sd[name].data[trg_idx].copy_(s.data[src_idx])
                    trg_ptr_after = t.data_ptr()
                    assert trg_ptr_b4 == trg_ptr_after, f"Data pointer changed for {prefix}"

        def copy_embedding_weights(key, trg_token2idx, src_token2idx):
            common_tokens = set(trg_token2idx.keys()).intersection(set(src_token2idx.keys()))

            if set(trg_token2idx.keys()) != set(src_token2idx.keys()):
                logger.warning(f"WARNING: Tokens for {key} are not the same. Source has {len(src_token2idx)} tokens and target has {len(trg_token2idx)} tokens. Copying {len(common_tokens)} common tokens.")

            token_idx_mapping = {trg_token2idx[token]: src_token2idx[token] for token in common_tokens}
            copy_(key, token_idx_mapping)


        copy_embedding_weights('wte.', trg_grid_token2idx, src_grid_token2idx)

        # copy program embeddings
        copy_embedding_weights('pte.', trg_prog_token2idx, src_prog_token2idx)

        copy_('inp_inject.')

        if config_trg.n_layer < config_src.n_layer:
            logger.warning(f"WARNING: Number of blocks in target model is less than source model. Copying only the first {config_trg.n_layer} blocks")

        # Copy transformer blocks
        for block_idx in range(config_trg.n_layer):
            trg_block_key = f'blocks.{block_idx}'

            if block_idx >= config_src.n_layer:
                logger.warning(f"WARNING: Copying block {block_idx} from the last block of the source model")
                src_block_idx = config_src.n_layer - 1
            else:
                src_block_idx = block_idx

            src_block_key = f'blocks.{src_block_idx}'
            copy_(f'{trg_block_key}.', src_prefix=f'{src_block_key}.')

        # ln_f
        copy_('ln_f.')

        # lm_head
        copy_('lm_head.')
        
#%%


# %%
