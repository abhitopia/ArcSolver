#%%
import math
from typing import List, Tuple
import torch
from src.solver import create_solver
from src.tokenizer import MODEL_INPUT, MODEL_OUTPUT, GridTokenizer
import torch.nn.functional as F
def create_y_t(y, t_start, t_end):
    y_t = MODEL_OUTPUT(
        grid=y.grid[:, t_start:t_end],
        grid_indices=y.grid_indices[:, t_start:t_end, :],
        target_grid=y.target_grid[:, t_start:t_end],
    )
    return y_t
data = torch.load('/Users/abhishekaggarwal/synced_repos/ArcSolver/task_1a6449f1.pt')
# %%
x, y = data['example']
# %%
ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
solver = create_solver(ckt_path,
                lr=0.01,
                jit=False,
                save=True)
# %%
model = solver.model
# %%
model.load_state_dict(data['model_state_dict'])
# %%
iter_logits, _ = model(x, y)
logits = iter_logits
_, predicted_tokens = torch.max(logits, dim=2)
predicted_tokens = [11] + predicted_tokens[0].tolist()[:-1]
target_tokens = y.grid.tolist()[0]
# %%
predicted_tokens == target_tokens
# %%
gp, gs = model.greedy_search(x.grid[0].tolist(), x.grid_indices[0].tolist())
# %%
gp == target_tokens
# %%
target_tokens
# %%

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

    print("X", x)

    _, cache = self.forward(
            x=x,
            y=None,
            return_cache=True
    )

    kv_cache, enc_mask, dec_mask = cache
    print("Initial Cache", print_cache(kv_cache))

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
    return output_list, math.exp(output_log_prob)
    

gp, gs = greedy_search(model, x.grid[0].tolist(), x.grid_indices[0].tolist(), max_length=2)

# %%

def print_cache(cache):
    for i, iter_cache in enumerate(cache):
        for b, block in enumerate(iter_cache):
            print(i, b, "k", block[0].sum().item(), "v", block[1].sum().item())

def incremental_forward(model, x, y):
    print("X", x)
    _, cache = model(x, None, return_cache=True)
    kv_cache, enc_mask, dec_mask = cache

    print("Initial Cache", print_cache(kv_cache))
    len_dec = y.grid.size(1)
    return
    logits_out = []
    for i in range(len_dec):
        y_i = MODEL_OUTPUT(grid=y.grid[:, i:i+1], grid_indices=y.grid_indices[:, i:i+1], target_grid=None)
        logits, cache = model.forward_inc(y_i, cache)
        logits_out.append(logits)

    logits_out = torch.cat(logits_out, dim=1)
    # loss = loss_fn(logits_out, y.target_grid)
    return logits_out

inc_forward = incremental_forward(model, x, y)
#%%
model.config
#%%
T = 1
next_y = create_y_t(y, 0, T)
slogits, _ = model.forward(x, y=next_y, return_cache=True)

# %%
_, cache = model.forward(x, y=None, return_cache=True)

# %%
tlogits, _ = model(x, y)
#%%
ilogits, cachen = model.forward_inc(next_y, cache=cache)
# %%
slogits.shape, tlogits.shape, ilogits.shape, inc_forward.shape
# %%    
slogits, ilogits, tlogits[:, T-1, :], inc_forward[:, T-1, :]
# %%
# enc_inp, enc_valid_mask, enc_indices = model.contruct_encoder_input(x)
# dec_inp, dec_valid_mask, dec_indices = model.contruct_decoder_input(next_y)
# # %%
# torch.all(enc_valid_mask), torch.all(cache[1])
# # %%
# enc_indices
# # %%
# enc_indices, dec_indices
# # %%
# dec_indices.shape
# # %%
# dec_inp.size(1)

# %%
x

# %%
inc_forward[:, T-1, :]
# %%
