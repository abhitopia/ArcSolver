#%%
import sys
from collections import OrderedDict

src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory


from src.lazy_adamw import LazyAdamW
import numpy as np
import torch
from src.tokenizer import MODEL_INPUT, MODEL_OUTPUT
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
# %%

import logging
logger = logging.getLogger(__name__)

src_config = REPLConfig(
        prog_vocab_size=15,
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
        prog_vocab_size=15,
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
