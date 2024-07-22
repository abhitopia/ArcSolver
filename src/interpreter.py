#%%
from dataclasses import dataclass
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from .dataset import ProgramTokenizer, GridTokenizer


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



class SelfAttention(nn.Module):
    def __init__(self, config: InterpreterConfig):
        super().__init__()
        self.config = config
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
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        ## attention (materializes the large (T,T) matrix for all the queries and keys)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.config.causal) # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


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
    def __init__(self, config: InterpreterConfig):
        super().__init__()

        self.mixers = nn.ModuleList()

        # Initialize shared_mixer to None
        shared_mixer = None

        # Instantiate the shared_mixer block once if sharing is enabled
        if config.share_mixer:
            shared_mixer = nn.Sequential(nn.LayerNorm(config.n_dim), SelfAttention(config))

        for _ in range(config.n_mixers):
            if config.share_mixer:
                self.mixers.append(shared_mixer)
            else:
                self.mixers(nn.Sequential(nn.LayerNorm(config.n_dim), SelfAttention(config)))

        self.normed_mlp = nn.Sequential(
                                nn.LayerNorm(config.n_dim),
                                MLP(config))
        

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
        self.blocks = nn.ModuleList([MixingBlock(config) for _ in range(config.n_blocks)])

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
        self.wpe = nn.Embedding(config.max_seq_len, config.n_dim)
        self.recurrent_block = RecurrentBlock(config)
        self.ln_f = nn.LayerNorm(config.n_dim)

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
        pos_emb = self.wpe(pos) # position embeddings of shape (T, n_embd)
        inp_emb = self.wte(inp_idx) # token embeddings of shape (B, T2, n_embd)
        inp_x = inp_emb + pos_emb

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
    

    def get_optimizer(self, 
                    model_lr,
                    model_weight_decay,
                    device_type,
                    prog_lr_scale=1.0,
                    prog_wd_scale=0.0):
        # Separate the embedding parameters
        program_params = [p for n, p in self.named_parameters() if 'pte' in n and p.requires_grad]
        model_params = [p for n, p in self.named_parameters() if 'pte' not in n and p.requires_grad]

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in model_params if p.dim() >= 2]
        nodecay_params = [p for p in model_params if p.dim() < 2]
        optim_groups = [
            {'params': program_params,
             'lr': model_lr * prog_lr_scale,
              'weight_decay': model_weight_decay * prog_wd_scale},
            {'params': decay_params,
              'lr': model_lr,
              'weight_decay': model_weight_decay},
            {'params': nodecay_params,
              'lr': model_lr,
              'weight_decay': 0.0}]
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=model_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


    def check_compatible(self, other: "Interpreter", compare_prog_vocab=True) -> bool:
        assert isinstance(other, Interpreter), "Can only compare with another Interpreter instance"
        
        config = self.config
        other_config = other.config


        or_conditions = [
            config.n_mixers == 1 and other_config.n_mixers == 1,
            config.share_mixer == other_config.share_mixer,
            config.share_mixer and other_config.n_mixers == 1,
        ]

        and_conditions = [
            self.prog_tokenizer == other.prog_tokenizer,
            self.grid_tokenizer == other.grid_tokenizer,
            config.max_seq_len <= other_config.max_seq_len,
            config.grid_vocab_size == other_config.grid_vocab_size,
            config.n_dim == other_config.n_dim,
            config.n_head == other_config.n_head,
            config.n_embd == other_config.n_embd,
            config.causal == other_config.causal,
            config.max_seq_len == other_config.max_seq_len,
            config.n_blocks == other_config.n_blocks,
            any(or_conditions)
        ]

        if compare_prog_vocab:
            and_conditions.append(config.prog_vocab_size == other_config.prog_vocab_size)

        return all(and_conditions)

        
#%%
# %%
