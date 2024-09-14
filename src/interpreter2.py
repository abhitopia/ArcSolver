#%%
from dataclasses import dataclass
import inspect
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import is_power_of_two, get_logger
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


    def forward(self, x, attn_mask=None, past_key_value=None):
        # x: (B, T, C)
        B, T, C = x.size()

        # qkv: (B, T, 3 * C)
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_dim, dim=2)

        # Reshape for multi-head attention, but do not transpose yet!
        q = q.view(B, T, self.n_head, C // self.n_head)  # (B, T, n_head, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)

        # If past_key_value is present, concatenate past keys and values BEFORE applying RoPE
        if past_key_value is not None:
            past_k, past_v = past_key_value  # K: (B, T_past, n_head, head_dim), V: (B, n_head, T_past, head_dim)
            k = torch.cat([past_k, k], dim=1)  # Concatenate along sequence length dimension
            v = torch.cat([past_v, v], dim=2)

        # Update present_key_value
        present_key_value = (k, v)

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

        return y, present_key_value



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

    def forward(self, x, attn_mask=None, past_key_value=None):
        attn_output, present_key_value = self.attn(self.rmsnorm(x), attn_mask=attn_mask, past_key_value=past_key_value)
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.normed_mlp(x))
        return x, present_key_value


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
    def get_attn_mask(src_len, trg_len, non_causal_prefix_len):
        # Note: One may be prompted to default non_causal_prefix_len to None
        # but if you think further, you will realise that in case src_len < any(non_causal_prefix_len)
        # then the default value of None will not work. Hence, it is better to always pass the
        # non_causal_prefix_len explicitly

        batch_size = non_causal_prefix_len.size(0)
        device = non_causal_prefix_len.device

        # CREATE ATTENTION MASK Shape [bs, trg_len, src_len]
        offset = src_len - trg_len

        # print(f"Offset: {offset}")
        causal_mask = torch.ones(batch_size, trg_len, src_len, dtype=torch.bool, device=device).tril(diagonal=offset)

        # Create a range tensor for comparison
        idx = torch.arange(src_len, device=device).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, src_len]

        # Expand inp_lens to match the dimensions for broadcasting
        expanded_inp_lens = non_causal_prefix_len.unsqueeze(1).unsqueeze(2)  # Shape: [bs, 1, 1]

        # Create the full mask by using broadcasting
        non_causal_mask = (idx < expanded_inp_lens).expand(-1, trg_len, -1)  # Shape: [bs, trg_len, src_len]

        # # Combine the lower triangular mask with the full mask
        attn_mask = causal_mask | non_causal_mask

        attn_mask = attn_mask.to(device) if device is not None else attn_mask

        # print("Sum of attn_mask", attn_mask.sum())
        return attn_mask.unsqueeze(1)  # Shape: [bs, 1, trg_len, src_len]


    
    def forward(self, prog_idx, inp_idx, inp_len, n_loops, next_input_idx=None, max_grad_loops=None, past_key_values=None):
        # idx is of shape (B, T)

        if prog_idx is None or inp_idx is None:
            assert prog_idx is None and inp_idx is None, "Both program and input must be None"
            assert next_input_idx is not None, "next_input_idx must be provided if program and input are None"
            assert past_key_values is not None, "past_key_values must be provided if program and input are None"
            assert len(past_key_values) == n_loops, "Number of past_key_values count must match number of loops"
            assert len(past_key_values[0]) == self.config.n_layer, "Number of past_key_values blocks must match number of layers"
            past_key = past_key_values[0][0][0]
            T_past = past_key.size(1)
            T_future = next_input_idx.size(1)
            batch_seq_len = T_past + T_future
            B1 = past_key.size(0)
            B2 = next_input_idx.size(0)
            assert B1 == B2, "Batch size of past_key_values and next_input_idx must match"
            assert max_grad_loops is None, "max_grad_loops must be None if program and input are None"

        else:
            assert next_input_idx is None, "next_input_idx must be None if program and input are provided"
            assert past_key_values is None, "past_key_values must be None if program and input are provided"

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



        if next_input_idx is None:
            # forward the token and position embeddings
            prog_emb = self.pte(prog_idx)  # (B, T1, n_dim)
            inp_emb = self.wte(inp_idx)    # (B, T2, n_dim)
            x = torch.cat((prog_emb, inp_emb), dim=1)
        else:
            x = self.wte(next_input_idx)  # (B, T3, n_dim)

        attn_mask = self.get_attn_mask(src_len=batch_seq_len,  trg_len=x.size(1), non_causal_prefix_len=inp_len)

        output = torch.zeros_like(x)
        convergence_mse = []

        # Initialize past_key_values if not provided, with one set of kv-cache for each block and each loop
        if past_key_values is None:
            past_key_values = [[None] * len(self.blocks) for _ in range(n_loops)]

        # List to store the updated kv-cache for each loop
        updated_past_key_values = []

        # print(f"Interpreter.forward: x shape: {x.shape}, output shape: {output.shape}")

        for loop_id in range(n_loops):
            prev_output = output

            # Create a new list of past_key_values for the current loop
            loop_past_key_values = []

            with torch.set_grad_enabled(loop_id >= grad_loop_start and self.training):
                # Inject input and output from previous loop iteration
                layer_inp = torch.cat((x, output), dim=-1)  # (B, T, 2 * n_dim)
                output = self.inp_inject(layer_inp)         # (B, T, n_dim)

                # new_past_key_values = []
                for i, block in enumerate(self.blocks):
                    # Pass the past_key_values from the specific loop_id
                    output, present_key_value = block(output, attn_mask=attn_mask, past_key_value=past_key_values[loop_id][i])

                    # Store the updated kv-cache for the current block in the current loop
                    loop_past_key_values.append(present_key_value)


            # Update MSE to track convergence between loops
            output_mse = F.mse_loss(output, prev_output)
            convergence_mse.append(output_mse.item())

            # Store the updated kv-cache for this loop iteration
            updated_past_key_values.append(loop_past_key_values)

        # Forward the final layernorm and the classifier
        output = self.ln_f(output)
        logits = self.lm_head(output)  # (B, T, vocab_size)

        return logits, convergence_mse, updated_past_key_values


    def greedy_search(self, prog_idx, inp_idx, n_loops, max_length, eos_token_id=12):
        # Assume prog_idx and inp_idx are lists of integers
        # Batch size is 1

        device = next(self.parameters()).device  # Get the device (CPU or GPU) from the model parameters

        # Compute inp_len as len(prog_idx) + len(inp_idx)
        inp_len = torch.tensor([len(prog_idx) + len(inp_idx)], dtype=torch.long, device=device)  # Shape: (1,)

        # Convert prog_idx and inp_idx to tensors and add batch dimension
        prog_idx = torch.tensor([prog_idx], dtype=torch.long, device=device)  # Shape: (1, len(prog_idx))

        # Add EOS token to the end of the input sequence
        pad_token_id = self.grid_tokenizer.PAD_IDX
        inp_idx = torch.tensor([inp_idx + [pad_token_id]], dtype=torch.long, device=device)    # Shape: (1, len(inp_idx))

        # Compute inp_len
        # Initialize the output sequence and past_key_values
        output_sequence = torch.tensor([], dtype=torch.long, device=device)  # Shape: (seq_len,)
        past_key_values = None

        for t in range(max_length):
            # Prepare the next input (the last generated token)
            if output_sequence.size(0) > 0:
                last_token = output_sequence[-1].item()
                # If EOS token is generated, stop
                if last_token == eos_token_id:
                    break
                next_input_idx = torch.tensor([[last_token]], dtype=torch.long, device=device)  # Shape: (1, 1)
            else:
                next_input_idx = None  # At the first time step

            # Run the model forward
            logits, _, past_key_values = self.forward(
                prog_idx if next_input_idx is None else None,
                inp_idx if next_input_idx is None else None,
                inp_len,
                n_loops,
                next_input_idx=next_input_idx,
                past_key_values=past_key_values
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

    def beam_search(self, prog_idx, inp_idx, inp_len, n_loops, beam_width, max_length, eos_token_id):
        device = prog_idx.device
        B = prog_idx.size(0)

        # Initialize the beam with empty sequences, zero scores, and kv-cache for each loop/block
        beam = [[(torch.tensor([], dtype=torch.long, device=device), 0.0, [[None]*len(self.blocks) for _ in range(n_loops)], False)] for _ in range(B)]  # Added False to track completion
        completed_sequences = [[] for _ in range(B)]  # Store completed sequences separately

        for t in range(max_length):
            all_candidates = [[] for _ in range(B)]  # Store all candidates for each batch element

            for b in range(B):
                candidates = beam[b]
                print(f"Batch {b}: {len(candidates)} candidates")

                # Expand each candidate in the beam
                for seq, score, past_key_values, is_completed in candidates:
                    print(f"Sequence: {seq.tolist()}, Score: {score}")

                    # If the sequence is completed, move it to the completed list but keep it in the beam
                    if is_completed:
                        completed_sequences[b].append((seq, score))
                        all_candidates[b].append((seq, score, past_key_values, True))  # Keep in beam but mark completed
                        continue  # Don't expand this sequence further

                    # Prepare the next input (i.e., the last generated token in the sequence)
                    if len(seq) > 0:
                        output_idx = seq.unsqueeze(0)  # (1, seq_len)
                    else:
                        output_idx = None

                    # Run the forward pass using the past_key_values from this specific beam
                    logits, _, new_past_key_values = self.forward(
                        prog_idx[b].unsqueeze(0),
                        inp_idx[b].unsqueeze(0),
                        inp_len[b].unsqueeze(0),
                        n_loops,
                        next_input_idx=output_idx,
                        past_key_values=past_key_values
                    )

                    # Get the logits for the next token
                    next_logits = logits[:, -1, :]  # (1, vocab_size)
                    next_log_probs = F.log_softmax(next_logits, dim=-1)  # (1, vocab_size)

                    # Get the top beam_width tokens
                    topk_log_probs, topk_tokens = torch.topk(next_log_probs, beam_width)

                    # For each possible next token
                    for i in range(beam_width):
                        token = topk_tokens[0, i]
                        token_log_prob = topk_log_probs[0, i]
                        new_seq = torch.cat([seq, token.unsqueeze(0)])
                        new_score = score + token_log_prob.item()

                        # Clone the past_key_values for the new candidate (so they don't share references)
                        cloned_past_key_values = [[(k.clone(), v.clone()) if k is not None else None for (k, v, *_) in loop_kv] for loop_kv in new_past_key_values]

                        # Check if the sequence is completed (EOS token found)
                        is_completed = token.item() == eos_token_id

                        # Add the new sequence, score, kv-cache, and completion status to the candidate list for batch b
                        all_candidates[b].append((new_seq, new_score, cloned_past_key_values, is_completed))

            # Reorder and select the top beam_width sequences for each batch element
            for b in range(B):
                candidates = all_candidates[b]
                ordered = sorted(candidates, key=lambda tup: tup[1], reverse=True)  # Sort by score (descending)
                
                # Keep only top beam_width candidates, including both completed and non-completed sequences
                beam[b] = ordered[:beam_width]

            # Early stopping if all sequences are completed
            if all(len(completed_sequences[b]) >= beam_width for b in range(B)):
                break

        # After the loop ends, gather all completed sequences along with remaining beam candidates
        final_sequences = []
        for b in range(B):
            # Return all completed sequences and, if there are no completed sequences, return the top beam candidates
            final_sequences.append(completed_sequences[b] if completed_sequences[b] else beam[b])

        return final_sequences

    # def beam_search(self, prog_idx, inp_idx, inp_len, n_loops, beam_width, max_length, eos_token_id):
    #     device = prog_idx.device
    #     B = prog_idx.size(0)

    #     # Initialize the beam with empty sequences, zero scores, and kv-cache for each loop/block
    #     beam = [[(torch.tensor([], dtype=torch.long, device=device), 0.0, [[None]*len(self.blocks) for _ in range(n_loops)])] for _ in range(B)]
    #     completed_sequences = [[] for _ in range(B)]

    #     for t in range(max_length):
    #         all_candidates = [[] for _ in range(B)]  # Store all candidates for each batch element
    #         for b in range(B):
    #             candidates = beam[b]
    #             # print(f"Batch {b}: {len(candidates)} candidates")

    #             # Expand each candidate in the beam
    #             for seq, score, past_key_values in candidates:
    #                 # print(f"Sequence: {seq.tolist()}, Score: {score}")

    #                 # If the sequence is complete (EOS token found), add it to completed sequences
    #                 if len(seq) > 0 and seq[-1].item() == eos_token_id:
    #                     completed_sequences[b].append((seq, score))
    #                     continue

    #                 # Prepare the next input (i.e., the last generated token in the sequence)
    #                 if len(seq) > 0:
    #                     output_idx = seq.unsqueeze(0)  # (1, seq_len)
    #                 else:
    #                     output_idx = None

    #                 # Run the forward pass using the past_key_values from this specific beam
    #                 logits, _, new_past_key_values = self.forward(
    #                     prog_idx[b].unsqueeze(0),
    #                     inp_idx[b].unsqueeze(0),
    #                     inp_len[b].unsqueeze(0),
    #                     n_loops,
    #                     next_input_idx=output_idx,
    #                     past_key_values=past_key_values
    #                 )

    #                 # Get the logits for the next token
    #                 next_logits = logits[:, -1, :]  # (1, vocab_size)
    #                 next_log_probs = F.log_softmax(next_logits, dim=-1)  # (1, vocab_size)

    #                 # Get the top beam_width tokens
    #                 topk_log_probs, topk_tokens = torch.topk(next_log_probs, beam_width)

    #                 # For each possible next token
    #                 for i in range(beam_width):
    #                     token = topk_tokens[0, i]
    #                     token_log_prob = topk_log_probs[0, i]
    #                     new_seq = torch.cat([seq, token.unsqueeze(0)])
    #                     new_score = score + token_log_prob.item()


    #                     # Clone the past_key_values for the new candidate (so they don't share references)
    #                     cloned_past_key_values = [[(k.clone(), v.clone()) if k is not None else None for k, v in loop_kv] for loop_kv in new_past_key_values]

    #                     # Add the new sequence, score, and kv-cache to the candidate list for batch b
    #                     all_candidates[b].append((new_seq, new_score, cloned_past_key_values))

    #         # Reorder and select the top beam_width sequences for each batch element
    #         for b in range(B):
    #             candidates = all_candidates[b]
    #             ordered = sorted(candidates, key=lambda tup: tup[1], reverse=True)  # Sort by score (descending)
    #             beam[b] = ordered[:beam_width]  # Keep only top beam_width candidates

    #         # Early stopping if all sequences are completed
    #         if all(len(completed_sequences[b]) >= beam_width for b in range(B)):
    #             break

    #     # Collect all completed sequences, and if there are no completed sequences, use partial ones
    #     final_sequences = []
    #     for b in range(B):
    #         # If there are completed sequences, return them all
    #         if completed_sequences[b]:
    #             final_sequences.append(completed_sequences[b])
    #         # Otherwise, return the top beam candidates (even if incomplete)
    #         else:
    #             final_sequences.append(beam[b])

    #     return final_sequences



    # # Beam Search Function
    # def beam_search(self, prog_idx, inp_idx, inp_len, n_loops, beam_width, max_length, eos_token_id):
    #     device = prog_idx.device
    #     B = prog_idx.size(0)
    #     # Initialize the beam with empty sequences and zero scores
    #     beam = [[(torch.tensor([], dtype=torch.long, device=device), 0.0)] for _ in range(B)]
    #     completed_sequences = [[] for _ in range(B)]

    #     for t in range(max_length):
    #         all_candidates = [[] for _ in range(B)]
    #         for b in range(B):
    #             candidates = beam[b]
    #             for seq, score in candidates:
    #                 if len(seq) > 0 and seq[-1].item() == eos_token_id:
    #                     # Sequence is already completed
    #                     completed_sequences[b].append((seq, score))
    #                     continue
    #                 # Prepare the output_idx
    #                 if len(seq) > 0:
    #                     output_idx = seq.unsqueeze(0)  # (1, seq_len)
    #                 else:
    #                     output_idx = None
    #                 # Run the model forward
    #                 logits, _ = self.forward(
    #                     prog_idx[b].unsqueeze(0),
    #                     inp_idx[b].unsqueeze(0),
    #                     inp_len,
    #                     n_loops,
    #                     next_input_idx=output_idx
    #                 )
    #                 # Get the logits for the next token
    #                 next_logits = logits[:, -1, :]  # (1, vocab_size)
    #                 next_log_probs = F.log_softmax(next_logits, dim=-1)  # (1, vocab_size)
    #                 # Get the top beam_width tokens
    #                 topk_log_probs, topk_tokens = torch.topk(next_log_probs, beam_width)
    #                 # For each possible next token
    #                 for i in range(beam_width):
    #                     token = topk_tokens[0, i]
    #                     token_log_prob = topk_log_probs[0, i]
    #                     new_seq = torch.cat([seq, token.unsqueeze(0)])
    #                     new_score = score + token_log_prob.item()
    #                     all_candidates[b].append((new_seq, new_score))
    #         # Reorder and select the top beam_width sequences for each batch element
    #         for b in range(B):
    #             candidates = all_candidates[b]
    #             ordered = sorted(candidates, key=lambda tup: tup[1], reverse=True)
    #             beam[b] = ordered[:beam_width]
    #     # Collect the best sequences
    #     final_sequences = []
    #     for b in range(B):
    #         sequences = completed_sequences[b] if completed_sequences[b] else beam[b]
    #         best_seq = max(sequences, key=lambda tup: tup[1])[0]
    #         final_sequences.append(best_seq)
    #     return final_sequences
 

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
