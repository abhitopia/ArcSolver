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

logger = get_logger()

@dataclass
class InterpreterConfig:
    prog_vocab_size: int # number of program tokens
    grid_vocab_size: int  # number of array element tokens (one extra for niceness)
    n_dim: int # dimension of the model
    n_head: int # number of heads within each self-attention block
    n_mixers: int # number of self-attention layers within each transformer block
    n_blocks: int # number of transformer blocks within each recurrence block
    n_rec_layers: int # number of recurrences
    n_embd: int = -1 # embedding dimension (defaults to n_dim )
    share_mixer: bool = True # Whether to tie the weights of the mixing layers within a transformer block
    causal: bool = False # whether to use causal attention
    max_seq_len: int = 1024 # max sequence length


    def __post_init__(self):
        assert is_power_of_two(self.prog_vocab_size), "Program vocab size must be a power of 2"
        assert is_power_of_two(self.grid_vocab_size), "Grid vocab size must be a power of 2"
        assert is_power_of_two(self.n_dim), "Model dimension must be a power of 2"

        if self.n_dim % self.n_head != 0:
            raise ValueError("n_dim must be divisible by n_head")

        if self.n_embd <= 0:
            self.n_embd = self.n_dim

    def to_dict(self):
        return {
            'max_seq_len': self.max_seq_len,
            'prog_vocab_size': self.prog_vocab_size,
            'grid_vocab_size': self.grid_vocab_size,
            'n_blocks': self.n_blocks,
            'n_mixers': self.n_mixers,
            'share_mixer': self.share_mixer,
            'n_head': self.n_head,
            'n_dim': self.n_dim,
            'n_embd': self.n_embd,
            'n_rec_layers': self.n_rec_layers,
            'causal': self.causal
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
        self.c_attn = nn.Linear(config.n_dim, 3 * config.n_dim)
        # output projection
        self.c_proj = nn.Linear(config.n_dim, config.n_dim)
        self.c_proj.RESCALE_RESIDUAL = 1

        # regularization
        self.n_head = config.n_head
        self.n_dim = config.n_dim


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
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
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.config.causal) # flash attention

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


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_dim, 4 * config.n_dim)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_dim, config.n_dim)
        self.c_proj.RESCALE_RESIDUAL = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class MixingBlock(nn.Module):
    def __init__(self, config: InterpreterConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()

        self.mixers = nn.ModuleList()

        # Initialize shared_mixer to None
        shared_mixer = None

        # Instantiate the shared_mixer block once if sharing is enabled
        if config.share_mixer:
            shared_mixer = nn.Sequential(RMSNorm(config.n_dim), SelfAttention(config, rope))

        for _ in range(config.n_mixers):
            if config.share_mixer:
                self.mixers.append(shared_mixer)
            else:
                self.mixers.append(nn.Sequential(RMSNorm(config.n_dim),
                                                SelfAttention(config)))

        self.normed_mlp = nn.Sequential(
                                RMSNorm(config.n_dim),
                                SwiGLUFFN(config.n_dim, 4 * config.n_dim))
        

    def forward(self, x):

        # Run the x through each mixer in the ModuleList
        for mixer in self.mixers:
            x = x + mixer(x)

        # Finally run the x through the normed MLP
        x = x + self.normed_mlp(x)
        return x


class RecurrentBlock(nn.Module):
    def __init__(self, config: InterpreterConfig):
        super().__init__()
        self.config = config
        self.rope = RotaryPositionalEmbeddings(config.n_dim, config.max_seq_len)
        self.blocks = nn.ModuleList([MixingBlock(config, rope=self.rope) for _ in range(config.n_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
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

        if config.n_embd == config.n_dim:
            self.pte = nn.Embedding(config.prog_vocab_size, config.n_dim)
        else:
            self.pte = nn.Sequential(
                        nn.Embedding(config.prog_vocab_size, config.n_embd),
                        nn.Linear(config.n_embd, config.n_dim, bias=False))

        self.wte = nn.Embedding(config.grid_vocab_size, config.n_dim)
        # self.wpe = nn.Embedding(config.max_seq_len, config.n_dim)

        self.recurrent_block = RecurrentBlock(config)
        self.ln_f = RMSNorm(config.n_dim)

        self.lm_head = nn.Linear(config.n_dim, config.grid_vocab_size, bias=False)
        # weight sharing scheme
        self.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            gain = 1.0
            if hasattr(module, 'RESCALE_RESIDUAL'):
                num_res = self.config.n_blocks * self.config.n_rec_layers
                res_each_block = (self.config.n_mixers + 1) # 1 for MLP
                gain = (res_each_block * num_res ) ** -0.5
            torch.nn.init.xavier_normal_(module.weight, gain=gain)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

 
    def forward(self, prog_idx, inp_idx):
        # idx is of shape (B, T)
        B1, T1 = prog_idx.size()
        B2, T2 = inp_idx.size()
        assert B1 == B2, "Batch size of program and input must match"
        assert T1+T2 <= self.config.max_seq_len, f"Cannot forward sequence of length {T1+T2}, max_seq_len is only {self.config.max_seq_len}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T2, dtype=torch.long, device=inp_idx.device) # shape (T)
        prog_emb = self.pte(prog_idx) # program embeddings of shape (B, T1, n_embd)
        # pos_emb = self.wpe(pos) # position embeddings of shape (T, n_embd)
        inp_emb = self.wte(inp_idx) # token embeddings of shape (B, T2, n_embd)
        # inp_x = inp_emb + pos_emb
        inp_x = inp_emb

        x = torch.cat((prog_emb, inp_x), dim=1)  # concatenate program and input embeddings (B, T1+ T2, n_embd)

        # forward the blocks of the transformer
        for _ in range(self.config.n_rec_layers):
            x = self.recurrent_block(x)
        # forward the final layernorm and the classifier
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    

    @staticmethod
    def loss_fn(logits, targets):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
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
        
        config = self.config
        other_config = other.config

        and_conditions = [
            (config.n_dim == other_config.n_dim, "Model dimension should be the same"),
            (config.n_head == other_config.n_head, "Number of heads should be the same"),
            (config.n_embd == other_config.n_embd, "Embedding dimension should be the same"),
        ]

        for cond, msg in and_conditions:
            assert cond, msg
        
        mixer_or_conditions = [
            config.n_mixers == 1 and other_config.n_mixers == 1,
            config.share_mixer == other_config.share_mixer,
            config.share_mixer and other_config.n_mixers == 1,
        ]

        strict_conditions = [
            (self.prog_tokenizer == other.prog_tokenizer, "Program tokenizers should be the same"),
            (self.grid_tokenizer == other.grid_tokenizer, "Grid tokenizers should be the same"),
            (config.grid_vocab_size == other_config.grid_vocab_size, "Grid vocab size should be the same"),
            (config.prog_vocab_size == other_config.prog_vocab_size, "Program vocab size should be the same"),
            (config.n_blocks == other_config.n_blocks, "Number of blocks should be the same"),
            (config.causal == other_config.causal, "Causal should be the same"),
            (config.max_seq_len == other_config.max_seq_len, "Max sequence length should be the same"),
            (mixer_or_conditions, "Mixer configuration not compatible")
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
            common_tokens = set(src_token2idx.keys()).intersection(set(src_token2idx.keys()))
            if len(common_tokens) != len(trg_token2idx):
                logger.warning(f"WARNING: Number of matching tokens for {key} is {len(common_tokens)} but trg_vocab_size is {len(trg_token2idx)}")

            token_idx_mapping = {trg_token2idx[token]: src_token2idx[token] for token in common_tokens}
            copy_(key, token_idx_mapping)

        # Copy positional embeddings (wpe)
        max_seq_len = min(config_trg.max_seq_len, config_src.max_seq_len)
        copy_('wpe.', {i: i for i in range(max_seq_len)})

        # wte and lm_head (tied weights)
        copy_embedding_weights('wte.weight', trg_grid_token2idx, src_grid_token2idx)
        assert trg_sd['wte.weight'].data_ptr() == trg_sd['lm_head.weight'].data_ptr(), "wte and lm_head should be tied"
        assert src_sd['wte.weight'].data_ptr() == src_sd['lm_head.weight'].data_ptr(), "wte and lm_head should be tied"


        # copy program embeddings
        copy_embedding_weights('pte.', trg_prog_token2idx, src_prog_token2idx)

        if config_trg.n_blocks < config_src.n_blocks:
            logger.warning(f"WARNING: Number of blocks in target model is less than source model. Copying only the first {config_trg.n_blocks} blocks")

        # Copy mixer blocks
        for block_idx in range(config_trg.n_blocks):
            trg_block_key = f'recurrent_block.blocks.{block_idx}'

            if block_idx >= config_src.n_blocks:
                logger.warning(f"WARNING: Copying block {block_idx} from the last block of the source model")
                src_block_idx = config_src.n_blocks - 1
            else:
                src_block_idx = block_idx

            src_block_key = f'recurrent_block.blocks.{src_block_idx}'

            copy_(f'{trg_block_key}.normed_mlp.', src_prefix=f'{src_block_key}.normed_mlp.')

            if config_trg.n_mixers < config_src.n_mixers:
                logger.warning(f"WARNING: Number of mixers in target model is less than source model. Copying only the first {config_trg.n_mixers} mixers")

            for i in range(config_trg.n_mixers):
                    if i < config_src.n_mixers:
                        copy_(f'{trg_block_key}.mixers.{i}.', src_prefix=f'{src_block_key}.mixers.{i}.')
                    else:
                        # Otherwise just copy the last mixer
                        logger.warning(f"WARNING: Copying mixer {i} from the last mixer of the source model")
                        copy_(f'{trg_block_key}.mixers.{i}.', src_prefix=f'{src_block_idx}.mixers.{config_src.n_mixers-1}.')
        # ln_f
        copy_('ln_f.')
        
#%%


# %%
