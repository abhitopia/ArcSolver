#%%
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

config = REPLConfig(
        prog_vocab_size=15,
        n_dim=128,
        n_embd=16, 
        n_head=8,
        n_layer=3, 
        pnorm=2.0, 
        dropout=0.0,
        n_state_layer=3,
    )


x, y = create_test_inp(
    bs=3,
    inp_seq_len=4, 
    out_seq_len=4,
    grid_vocab_size=config.grid_vocab_size,
    tform_vocab_size=config.tform_vocab_size,
    perm_vocab_size=config.perm_vocab_size,
    prog_vocab_size=config.prog_vocab_size)

x.grid, x.grid_indices, y.grid, y.grid_indices

valid_mask = torch.cat([(x.grid != 0), (y.grid != 0)], dim=1)
print("Valid Mask", valid_mask)
model = REPL(config)

# scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model))
scripted_model = torch.jit.script(model)

#%%
%%time
logits, cache = model(x, y, iters=4, return_cache=True)
loss = model.compute_loss(logits, y)
logits[-1][:, :, 0], loss


#%%
%%time
logits, cache = scripted_model(x, y, iters=4, return_cache=True)
loss_scripted = scripted_model.compute_loss(logits, y)

# assert torch.allclose(loss, loss_scripted), f"Loss mismatch: {loss} != {loss_scripted}"
# logits[-1][:, :, 0], loss_scripted
#%%
def incremental_forward(model, x, y, iters=4):
    _, cache = model(x, None, iters=iters, return_cache=True)

    len_dec = y.grid.size(1)

    logits_out = [[] for _ in range(iters)]
    for i in range(len_dec):
        y_i = MODEL_OUTPUT(grid=y.grid[:, i:i+1], grid_indices=y.grid_indices[:, i:i+1], target_grid=None)
        logits, cache = model.forward_inc(y_i, cache, iters=iters)
        for j in range(iters):
            logits_out[j].append(logits[j])

    logits_out = [torch.cat(i, dim=1) for i in logits_out]
    loss = model.compute_loss(logits_out, y)
    return logits_out, loss


logits_inc, loss = incremental_forward(model, x, y, iters=4)
logits_inc[-1][:, :, 0], loss

#%%
logits_inc, loss_scripted = incremental_forward(scripted_model, x, y, iters=4)
logits_inc[-1][:, :, 0], loss_scripted
#%%

logits, cache = model(x, y, iters=4, return_cache=True)
loss = model.compute_loss(logits, y)
logits[-1][:, :, 0], loss
#%%
logits, cache = model(x, None, iters=4, return_cache=True)
logits[-1][:, :, 0], y.grid.shape

#%%
#%%
logits_next, cache_next = model.forward_inc(y, cache, iters=4)
loss_next = model.compute_loss(logits_next, y)
logits_next[-1][:, :, 0], y.grid.shape, loss_next
#%%

#%%
global_nan_value = 0.0
logits, cache = model(x, y, iters=4, return_cache=True)# %%
logits[-1][:, :, 0]
# logitsx[0]
# %%
global_nan_value = 0.0
logits = model.forwardb(x, y, iters=4)
logits[-1][:, :, 0]
#
#%%
len(cache[0][0]), cache[0][0][1].size()
# %%
# %%
%%time
predicted, score = model.greedy_search(
                            prog_idx=x.program[0].item(),
                            input_grid=x.grid[0, :].tolist(), 
                            input_indices=x.grid_indices[0, :].tolist(), 
                            iters=4,
                            color_perm_idx=x.color_permutation[0].item(),
                            array_tform_idx=x.array_transform[0].item(),
                        )

# %%
%%time
predicted, score_scripted = scripted_model.greedy_search(
                            prog_idx=x.program[0].item(),
                            input_grid=x.grid[0, :].tolist(), 
                            input_indices=x.grid_indices[0, :].tolist(), 
                            iters=4,
                            color_perm_idx=x.color_permutation[0].item(),
                            array_tform_idx=x.array_transform[0].item(),
                        )
score, score_scripted
# %%
