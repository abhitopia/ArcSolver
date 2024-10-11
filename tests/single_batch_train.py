#%%
from src.multilevel_loss import MultiLevelLoss, exp_spacing
from src.utils import map_to_tensors
import numpy as np
import torch
from src.tokenizer import MODEL_INPUT, MODEL_OUTPUT
from src.repl import REPLConfig, REPL

device = 'cpu'
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
        pnorm=1.0, 
        dropout=0.0,
    )


x, y = create_test_inp(
    bs=100,
    inp_seq_len=50, 
    out_seq_len=50,
    grid_vocab_size=config.grid_vocab_size,
    tform_vocab_size=config.tform_vocab_size,
    perm_vocab_size=config.perm_vocab_size,
    prog_vocab_size=config.prog_vocab_size)


x = map_to_tensors(x, lambda x: x.to(device))
y = map_to_tensors(y, lambda y: y.to(device))
## LOSS

#%%
model = REPL(config)
model = torch.jit.script(model)
model.to(device)
count_parameters_detailed(model)
loss_fn = MultiLevelLoss(
            pad_idx=0,
            edr=2.0,
            min_pct=0.4)
# %%

from torch.optim.adam import Adam

optimizer = Adam(model.parameters(), lr=1e-3)
#%%
import time
start_time = time.time()
num_iters = 10
for i in range(num_iters):
    optimizer.zero_grad()
    iter_logits, _ = model(x, y)
    loss = loss_fn(iter_logits, y.target_grid)
    loss.backward()
    optimizer.step()
    print(f"Loss: {loss.item()}")

print(f"Time taken: {(time.time() - start_time)/num_iters*1000} ms")
# %%

count_parameters_detailed(model)
# %%
# from src.multilevel_loss import MultiLevelLoss, exp_spacing
import numpy as np

def exp_spacing(
    n: int,
    rate: float = 1.0,
    min_val: float = 0.4,
    max_val: float = 1.0,
    threshold: float = 100.0
) -> np.ndarray:
    """
    Generates n points in the range [min_val, max_val] with spacing controlled by rate.

    Parameters:
    - n (int): Number of points to generate. Must be >= 1.
    - rate (float): Controls the spacing between points.
        - rate = 0: Uniform spacing.
        - rate > 0: More points near min_val.
        - rate < 0: More points near max_val.
    - min_val (float): Minimum value of the range.
    - max_val (float): Maximum value of the range.
    - threshold (float): Absolute rate beyond which it's treated as +inf or -inf.

    Returns:
    - np.ndarray: Array of n points with specified spacing.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if n == 1:
        return np.array([min_val])
    if n == 2:
        return np.array([min_val, max_val])
    
    # Define threshold to prevent overflow
    threshold = threshold  # You can adjust this value as needed

    # Handle extremely large positive or negative rates
    if rate >= threshold:
        # All points except the first are max_val
        points = np.full(n, max_val)
        points[0] = min_val
        return points
    elif rate <= -threshold:
        # All points except the last are min_val
        points = np.full(n, min_val)
        points[-1] = max_val
        return points

    # Generate uniform parameter t
    t = np.linspace(0, 1, n)

    if rate == 0:
        p = t
    else:
        # Compute exponentials safely
        try:
            numerator = 1 - np.exp(-rate * t)
            denominator = 1 - np.exp(-rate)
            p = numerator / denominator
        except OverflowError:
            # Fallback to extreme cases if overflow occurs
            if rate > 0:
                points = np.full(n, max_val)
                points[0] = min_val
                return points
            else:
                points = np.full(n, min_val)
                points[-1] = max_val
                return points

    # Map p from [0, 1] to [min_val, max_val]
    points = min_val + p * (max_val - min_val)

    # Handle potential numerical issues near the boundaries
    points = np.clip(points, min_val, max_val)

    return points

exp_spacing(8, 1.5, 0.0)
# %%
import torch
def exp_spacing_torch(
    n: int,
    rate: float = 0,
    min_val: float = 0.0,
    max_val: float = 1.0,
    threshold: float = 100.0,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generates n points in the range [min_val, max_val] with spacing controlled by rate.

    Parameters:
    - n (int): Number of points to generate. Must be >= 1.
    - rate (float): Controls the spacing between points.
        - rate = 0: Uniform spacing.
        - rate > 0: More points near min_val.
        - rate < 0: More points near max_val.
    - min_val (float): Minimum value of the range.
    - max_val (float): Maximum value of the range.
    - threshold (float): Absolute rate beyond which it's treated as +inf or -inf.
    - device (torch.device): Device on which to perform computations.
    - dtype (torch.dtype): Data type of the output tensor.

    Returns:
    - torch.Tensor: Tensor of n points with specified spacing.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if n == 1:
        return torch.tensor([min_val], device=device, dtype=dtype)
    if n == 2:
        return torch.tensor([min_val, max_val], device=device, dtype=dtype)
    
    # Handle extremely large positive or negative rates
    if rate >= threshold:
        # All points except the first are max_val
        points = torch.full((n,), max_val, device=device, dtype=dtype)
        points[0] = min_val
        return points
    elif rate <= -threshold:
        # All points except the last are min_val
        points = torch.full((n,), min_val, device=device, dtype=dtype)
        points[-1] = max_val
        return points
    
    # Generate uniform parameter t
    t = torch.linspace(0, 1, steps=n, device=device, dtype=dtype)
    
    if rate == 0:
        p = t
    else:
        # Convert rate to a tensor
        rate_tensor = torch.tensor(rate, device=device, dtype=dtype)
        
        # Compute exponentials safely
        numerator = 1 - torch.exp(-rate_tensor * t)
        denominator = 1 - torch.exp(-rate_tensor)
        
        # Handle potential division by zero when rate is very close to 0
        epsilon = 1e-10
        near_zero = torch.abs(denominator) < epsilon
        if near_zero.any():
            # Replace denominator with 1 to prevent division by zero; p will be t where denominator is near zero
            denominator = torch.where(near_zero, torch.ones_like(denominator), denominator)
            p = numerator / denominator
            # Where denominator was near zero, set p to t
            p = torch.where(near_zero, t, p)
        else:
            p = numerator / denominator
    
    # Map p from [0, 1] to [min_val, max_val]
    points = min_val + p * (max_val - min_val)
    
    # Ensure points are within [min_val, max_val]
    points = torch.clamp(points, min=min_val, max=max_val)
    
    return points

exp_spacing_torch(8, -100, 0.0)

# %%
