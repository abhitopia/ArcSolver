#%%
import random
import sys
from collections import OrderedDict

src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory


from src.lazy_adamw import LazyAdamW
import numpy as np
import torch
from src.tokenizer import MODEL_INPUT, MODEL_OUTPUT, ProgramTokenizer
from src.repl import REPLConfig, REPL

def create_test_inp(bs=10, inp_seq_len=10, out_seq_len=5, prog_vocab_size=15, perm_vocab_size=10, tform_vocab_size=11, grid_vocab_size=16, pad_idx=0, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
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

    target_grid = torch.cat([output_grid[:, 1:],
                            torch.full((bs, 1), pad_idx, dtype=output_grid.dtype)], dim=1)

    x = MODEL_INPUT(program=program,
                color_permutation=color_permutation,
                array_transform=array_transform, 
                grid=input_grid, 
                grid_indices=inp_indices,
                meta=None)
    y = MODEL_OUTPUT(grid=output_grid,
                grid_indices=out_indices,
                target_grid=target_grid
                )
    return x, y


# def load_prog_embeddings(self, trg_token2idx, src_state_dict, src_token2idx):
#     src_sd = src_state_dict
#     trg_sd = self.state_dict()

#     @torch.no_grad()
#     def copy_(prefix, idx_mapping=None, src_prefix=None):
#         for name, t in trg_sd.items():
#             if name.startswith(prefix):
#                 suffix = name[len(prefix):]
#                 src_name = src_prefix + suffix if src_prefix is not None else name
#                 s = src_sd[src_name]
#                 trg_ptr_b4 = t.data_ptr()
#                 if idx_mapping is None:
#                     t.data.copy_(s)
#                 else:
#                     for trg_idx, src_idx in idx_mapping.items():
#                         t.data[trg_idx].copy_(s.data[src_idx])

#                 trg_ptr_after = t.data_ptr()
#                 assert trg_ptr_b4 == trg_ptr_after, f"Data pointer changed for {prefix}"

#     @torch.no_grad()
#     def copy_embedding_weights(key, trg_token2idx, src_token2idx):
#         common_tokens = set(trg_token2idx.keys()).intersection(set(src_token2idx.keys()))

#         if set(trg_token2idx.keys()) != set(src_token2idx.keys()):
#             logger.warning(f"WARNING: Tokens for {key} are not the same. Source has {len(src_token2idx)} tokens and target has {len(trg_token2idx)} tokens. Copying {len(common_tokens)} common tokens.")

#         token_idx_mapping = {trg_token2idx[token]: src_token2idx[token] for token in common_tokens}
#         copy_(key, token_idx_mapping)


#     copy_embedding_weights('pte.0.', trg_token2idx, src_token2idx)

#%%

import logging
logger = logging.getLogger(__name__)


prog_vocab_size = 15

prog_tokens = ['prog_{}'.format(i) for i in range(prog_vocab_size)]

random.shuffle(prog_tokens)
prog_tokens_src = prog_tokens 
# print(prog_tokens_src)

random.shuffle(prog_tokens)
prog_tokens_trg = prog_tokens + ['dont_use']

print(prog_tokens_trg)
tokenizer_trg = ProgramTokenizer()
tokenizer_trg.build(prog_tokens_trg)

print(prog_tokens_src)
tokenizer_src = ProgramTokenizer()
tokenizer_src.build(prog_tokens_src)

tokenizer_trg.token2idx, tokenizer_src.token2idx
#%%
src_config = REPLConfig(
        prog_vocab_size=len(prog_tokens_src),
        n_dim=128,
        n_embd=16, 
        n_head=8,
        n_layer=5, 
        pnorm=2.0, 
        dropout=0.0,
        n_state_layer=5,
    )

src_model = REPL(src_config)

trg_config = REPLConfig(
        prog_vocab_size=len(prog_tokens_trg),
        n_dim=128,
        n_embd=16, 
        n_head=8,
        n_layer=5, 
        pnorm=2.0, 
        dropout=0.0,
        n_state_layer=5,
    )

trg_model = REPL(trg_config)


x, y = create_test_inp(
    bs=3,
    inp_seq_len=4, 
    out_seq_len=4,
    grid_vocab_size=trg_config.grid_vocab_size,
    tform_vocab_size=trg_config.tform_vocab_size,
    perm_vocab_size=trg_config.perm_vocab_size,
    prog_vocab_size=trg_config.prog_vocab_size)

x.grid, x.grid_indices, y.grid, y.grid_indices

#%%
logits, cache = src_model(x, y, iters=4, return_cache=True)
loss = src_model.compute_loss(logits, y)
logits[-1][:, :, 0], loss

#%%

trg_model.load_state_dict(src_model.state_dict(), strict=False)
trg_model.load_prog_embeddings( tokenizer_trg.token2idx, src_model.state_dict(), tokenizer_src.token2idx)

logits, cache = trg_model(x, y, iters=4, return_cache=False)
loss = trg_model.compute_loss(logits, y)
logits[-1][:, :, 0], loss
# %%



logits, cache = trg_model(x, y, iters=4, return_cache=True)
loss = trg_model.compute_loss(logits, y)
logits[-1][:, :, 0], loss

#%%
logits, cache = src_model(x, y, iters=4, return_cache=True)
loss = src_model.compute_loss(logits, y)
logits[-1][:, :, 0], loss

#%%
import re

trg_model.load_state_dict(src_model.state_dict(), strict=False)

logits, cache = trg_model(x, y, iters=4, return_cache=False)
loss = trg_model.compute_loss(logits, y)
logits[-1][:, :, 0], loss
# %%



# %%
