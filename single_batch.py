#%%
from src.multilevel_loss import MultiLevelLoss
from src.utils import map_to_tensors
import numpy as np
import torch
from src.tokenizer import MODEL_INPUT, MODEL_OUTPUT
from src.repl import REPLConfig, REPL

device = 'cuda'
def count_parameters_detailed(model):
    total_params = 0
    print("Parameter counts per layer:")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"{name}: {num_params} parameters")
    print(f"\nTotal parameters: {total_params}")



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



config = REPLConfig(
        prog_vocab_size=2000,
        n_dim=256,
        n_embd=16, 
        n_head=8,
        n_layer=2, 
        pnorm=2.0, 
        dropout=0.0,
    )


x, y = create_test_inp(
    bs=50,
    inp_seq_len=400, 
    out_seq_len=400,
    grid_vocab_size=config.grid_vocab_size,
    tform_vocab_size=config.tform_vocab_size,
    perm_vocab_size=config.perm_vocab_size,
    prog_vocab_size=config.prog_vocab_size)


x = map_to_tensors(x, lambda x: x.to(device))
y = map_to_tensors(y, lambda y: y.to(device))
## LOSS

#%%
model = REPL(config)
model.to(device)
model = torch.jit.script(model)
# model.to(device)

count_parameters_detailed(model)
loss_fn = MultiLevelLoss(
            pad_idx=0,
            edr=2.0,
            min_pct=0.4)
# %%

from torch.optim.adamw import AdamW
# from torch.optim.sgd import SGD

# optimizer = SGD(model.parameters(), lr=1e-3)
optimizer = AdamW(model.parameters(), lr=1e-3)

#%%
from src.task import DatasetLoader
from src.tokenizer import ArcTokenizer
from src.dataset import ArcExamplesDataset

training_data = DatasetLoader.TRAIN_ONLY.load(
    max_height=45,
    max_width=45,
    min_test=1,
    max_test=3,
    max_train=100,
    min_train=100,
)


train_examples = training_data.train_examples

tokenizer = ArcTokenizer()
tokenizer.build_program_tokenizer(train_examples)

train_ds = ArcExamplesDataset(train_examples, tokenizer)


train_dl = train_ds.get_dataloader(token_count=20000,
                                    pin_memory=False,
                                    shuffle=True,
                                    device=device,
                                    num_workers=0)

#%%
import time

for i, batch in enumerate(train_dl):
    x, y = batch
    x_new = MODEL_INPUT(
        program=x.program,
        color_permutation=x.color_permutation,
        array_transform = x.array_transform,
        grid = x.grid,
        grid_indices=x.grid_indices,
        meta=None
    )
    start_time = time.time()
    optimizer.zero_grad()
    iter_logits, _ = model(x_new, y)
    loss = loss_fn(iter_logits, y.target_grid)

    if i == 10:
        break



import torch.nn.functional as F
num_iters = 100
time_taken = 0
for i, batch in enumerate(train_dl):
    x, y = batch
    x_new = MODEL_INPUT(
        program=x.program,
        color_permutation=x.color_permutation,
        array_transform = x.array_transform,
        grid = x.grid,
        grid_indices=x.grid_indices,
        meta=None
    )
    start_time = time.time()
    optimizer.zero_grad()
    iter_logits, _ = model(x_new, y)
    loss = loss_fn(iter_logits, y.target_grid)

    # targets = y.target_grid
    # logits = iter_logits[-1]
    # loss =  F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none').view_as(targets)   # This will be a float tensor
    # mask = targets != 0
    # loss = (loss * mask).sum() / mask.sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    print(f"Loss: {loss.item()}")
    print(f"iter: {i}")
    end_time = time.time()

    time_taken += (end_time - start_time)
    if i == num_iters - 1:
        break

print("Time Taken(ms)", (time_taken*1000)/num_iters)

# #%%
# # %%
# # %%timeit ``
# # optimizer.zero_grad()
# iter_logits, _ = model(x, y)
# loss = loss_fn(iter_logits, y.target_grid)
# # loss.backward()
# # optimizer.step()
# # %%
# x.grid

# # %%
# iter_logits[-2][:, :, 1]
# # %%
# y.grid
# # %%

# %%
