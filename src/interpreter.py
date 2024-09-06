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
        assert is_power_of_two(self.n_dim), "Model dimension must be a power of 2"

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


    def forward(self, x, attn_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_dim)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_dim, dim=2)


        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # For Rope
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        k = self.rope(k).transpose(1, 2)
        q = self.rope(q).transpose(1, 2)

        ## attention (materializes the large (T,T) matrix for all the queries and keys)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.config.dropout) # flash attention
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.config.dropout) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


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


    def forward(self, x, attn_mask):
        x = x + self.dropout(self.attn(self.rmsnorm(x), attn_mask))
        x = x + self.dropout(self.normed_mlp(x))
        return x    



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

 
    def forward(self, prog_idx, inp_idx, inp_len, n_loops, max_grad_loops=None):
        # idx is of shape (B, T)
        B1, T1 = prog_idx.size()
        B2, T2 = inp_idx.size()
        output_len = T1 + T2
        assert output_len <= self.config.max_seq_len, f"Cannot forward sequence of length {T2}, max_seq_len is only {self.config.max_seq_len}"
        assert B1 == B2, "Batch size of program and input must match"
        assert T1 == 1, "Program input must be a single token"
        assert n_loops > 0, "Number of loops must be greater than 0"
        if max_grad_loops is None or max_grad_loops > n_loops: 
            max_grad_loops = n_loops
        assert 0 < max_grad_loops <= n_loops, "max_grad_loops must be less than or equal to n_loops and greater than 0"  

        grad_loop_start = n_loops - max_grad_loops
        attn_mask = torch.ones(output_len, output_len, dtype=torch.bool).tril(diagonal=0)
        attn_mask[:inp_len+1, :inp_len+1] = True
        attn_mask = attn_mask.to(prog_idx.device)

        # forward the token and posisition embeddings
        prog_emb = self.pte(prog_idx) # program embeddings of shape (B, T1, n_dim)
        inp_emb = self.wte(inp_idx) # token embeddings of shape (B, T2, n_dim)

        # x = prog_emb[..., None] * inp_emb[:, :, None, :]
        # x = x.reshape(B1, T2, -1)

        x = torch.cat((prog_emb, inp_emb), dim=1)
        output = torch.zeros_like(x)
        convergence_mse = []

        for loop_id in range(n_loops):
            # Only inject input 

            prev_output = output
            with torch.set_grad_enabled(loop_id >= grad_loop_start and self.training):
                layer_inp = torch.cat((x, output), dim=-1) # (B, T, 2*n_dim)
                output = self.inp_inject(layer_inp)  # (B, T, n_dim)
                for block in self.blocks:
                    output = block(output, attn_mask)

            output_mse = F.mse_loss(output, prev_output)
            convergence_mse.append(output_mse.item())

        # forward the final layernorm and the classifier
        output = self.ln_f(output)
        logits = self.lm_head(output) # (B, T, vocab_size)
        return logits, convergence_mse
    

    # def infer(self, prog_emb, inp_idx):
        
    #     inp_emb = self.wte(inp_idx) # token embeddings of shape (B, T2, n_dim)
    #     B2, T2 = inp_idx.size()

    #     x = prog_emb[..., None] * inp_emb[:, :, None, :]
    #     x = x.reshape(B2, T2, -1)

    #     for _ in range(self.config.n_rec_layer):
    #         for block in self.blocks:
    #             for _ in range(self.config.n_rec_block):
    #                 x = block(x)

    #     # forward the final layernorm and the classifier
    #     x = self.ln_f(x)
    #     logits = self.lm_head(x) # (B, T, vocab_size)
    #     return logits
    

    @staticmethod
    def loss_fn(logits, targets, l):
        logits_outs = logits[:, l:, :]
        targets_outs = targets[:, l:]
        loss =  F.cross_entropy(logits_outs.reshape(-1, logits_outs.size(-1)), targets_outs.reshape(-1))
        return loss
    
    def get_optimizer(
                self, 
                model_lr,
                prog_lr,
                model_wd,
                prog_wd=0.0,
                device_type=None,
            ):
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
