
import math
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import json
from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn

class Example(NamedTuple):
    input: Tensor
    output: Optional[Tensor]

class Task(NamedTuple):
    task_id: str
    train: List[Example]
    test: List[Example]

    def complexity(self)-> float:
        max_ratio = max([e.output.numel()/e.input.numel() for e in self.train])
        max_inp_size = max([e.input.numel() for e in self.test])
        max_test_output_size = max_ratio * max_inp_size
        ratio_test2train = len(self.test) / len(self.train)
        return max_test_output_size * ratio_test2train

class TaskSolution(NamedTuple):
    task_id: str
    predictions: List[List[Tensor]]
    scores: List[List[float]]
    log: Optional[List[Dict[str, Union[float, int]]]] = None

# class MODEL_INPUT(NamedTuple):
#     color_permutation: torch.Tensor
#     array_transform: torch.Tensor
#     program: torch.Tensor
#     grid: torch.Tensor
#     grid_indices: torch.Tensor
#     meta: Optional[List[Dict[str, str]]] = None

# class MODEL_OUTPUT(NamedTuple):
#     grid: torch.Tensor
#     grid_indices: torch.Tensor
#     target_grid: Optional[torch.Tensor] = None



class MODEL_INPUT(NamedTuple):
    is_inverse: torch.Tensor
    color_permutation: torch.Tensor
    array_transform: torch.Tensor
    program: torch.Tensor
    grid: torch.Tensor
    grid_indices: torch.Tensor
    meta: Optional[List[Dict[str, str]]] = None

    def unsqueeze(self, dim: int):
        return MODEL_INPUT(
            is_inverse=self.is_inverse.unsqueeze(dim),
            color_permutation=self.color_permutation.unsqueeze(dim),
            array_transform=self.array_transform.unsqueeze(dim),
            program=self.program.unsqueeze(dim),
            grid=self.grid.unsqueeze(dim),
            grid_indices=self.grid_indices.unsqueeze(dim),
            meta=self.meta
        )
    
    def squeeze(self, dim: int):
        return MODEL_INPUT(
            is_inverse=self.is_inverse.squeeze(dim),
            color_permutation=self.color_permutation.squeeze(dim),
            array_transform=self.array_transform.squeeze(dim),
            program=self.program.squeeze(dim),
            grid=self.grid.squeeze(dim),
            grid_indices=self.grid_indices.squeeze(dim),
            meta=self.meta
        )

class MODEL_OUTPUT(NamedTuple):
    grid: torch.Tensor
    grid_indices: torch.Tensor
    target_grid: Optional[torch.Tensor]
    
    def unsqueeze(self, dim: int):
        return MODEL_OUTPUT(
            grid=self.grid.unsqueeze(dim),
            grid_indices=self.grid_indices.unsqueeze(dim),
            target_grid=self.target_grid.unsqueeze(dim) if self.target_grid is not None else None
        )
    
    def squeeze(self, dim: int):
        return MODEL_OUTPUT(
            grid=self.grid.squeeze(dim),
            grid_indices=self.grid_indices.squeeze(dim),
            target_grid=self.target_grid.squeeze(dim) if self.target_grid is not None else None
        )


@torch.jit.script
def grid_indices(array: torch.Tensor) -> List[List[int]]:
    height = array.size(0)
    width = array.size(1)
    grid_height = height
    grid_width = width + 2

    rows = torch.arange(grid_height)
    cols = torch.arange(grid_width)
    rows_grid, cols_grid = torch.meshgrid(rows, cols, indexing='ij')
    indices = torch.stack((rows_grid, cols_grid), dim=2)
    indices = indices.reshape(-1, 2)

    indices_list = torch.jit.annotate(List[List[int]], [])
    for i in range(indices.size(0)):
        row = indices[i]
        indices_list.append([int(row[0].item()), int(row[1].item())])
    return indices_list

@torch.jit.script
def tokenize(string: str) -> List[int]:
    PAD_TOKEN = '<NOOP>'
    tokens = [PAD_TOKEN, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[[', ']', '[', ']]']
    token2idx = {token: idx for idx, token in enumerate(tokens)}
    # idx2token = {idx: token for idx, token in enumerate(tokens)}
    token_indices: List[int] = []
    for token in string.split():
        token_indices.append(token2idx[token])
    return token_indices

@torch.jit.script
def detokenize(token_indices: List[int]) -> str:
    PAD_TOKEN = '<NOOP>'
    tokens = [PAD_TOKEN, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[[', ']', '[', ']]']
    # token2idx = {token: idx for idx, token in enumerate(tokens)}
    idx2token: Dict[int, str] = {idx: token for idx, token in enumerate(tokens)}
    grid_str: List[str] = []
    for token in token_indices:
        grid_str.append(idx2token[token])
    result: str = ' '.join(grid_str)
    return result

@torch.jit.script
def serialize_array(array: Tensor) -> Tuple[Tensor, Tensor]:
    list_of_lists: List[List[int]] = array.tolist()
    array_str = str(list_of_lists)
    array_str = array_str.replace('],' , ' ],').replace('[[', '[[ ').replace(']]', ' ]]').replace(' [', ' [ ').replace(',', '')
    num_tokens = len(array_str.split())
    assert num_tokens == array.shape[0] * (array.shape[1] + 2)
    indices = grid_indices(array)
    assert num_tokens == len(indices)
    tokenized = tokenize(array_str)
    return torch.tensor(tokenized, device=array.device), torch.tensor(indices, device=array.device)

@torch.jit.script
def is_integer(s: str) -> bool:
    s = s.strip()
    if s == '':
        return False
    if s[0] == '-':
        return s[1:].isdigit()
    else:
        return s.isdigit()


@torch.jit.script
def deserialize_array(token_indices: List[int], device: str = 'cpu') -> Tensor:
    array_str = detokenize(token_indices)
    s = array_str.strip()
    
    # Remove outer brackets if present
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]

    # Replace '][' and '] [' with '];[' to prepare for splitting
    s = s.replace('][', '];[')
    s = s.replace('] [', '];[')

    # Split into sub-array strings
    sub_arrays_str = s.split(';')

    # Annotate 'result' as a list of lists of integers
    result = torch.jit.annotate(List[List[int]], [])
    for sub_array_str in sub_arrays_str:
        sub_array_str = sub_array_str.strip()
        
        # Remove brackets from sub-array if present
        if sub_array_str.startswith('[') and sub_array_str.endswith(']'):
            sub_array_str = sub_array_str[1:-1]

        # Split numbers by space
        num_strs = sub_array_str.strip().split(' ')

        # Annotate 'nums' as a list of integers
        nums = torch.jit.annotate(List[int], [])
        for num_str in num_strs:
            num_str = num_str.strip()
            if num_str != '':
                if is_integer(num_str):
                    nums.append(int(num_str))
                else:
                    # Print the invalid value and return an empty tensor
                    # print("Invalid integer value:", num_str)
                    return torch.empty((0, 0), dtype=torch.int64)
        
        result.append(nums)

    # If 'result' is empty, return an empty tensor
    if len(result) == 0:
        return torch.empty((0, 0), dtype=torch.int64)

    # Check that all sublists have the same length
    expected_length = len(result[0])
    for sublist in result:
        if len(sublist) != expected_length:
            # Return empty tensor if lengths are not uniform
            # print("Inconsistent sublist lengths.")
            return torch.empty((0, 0), dtype=torch.int64)

    # Convert each sublist to Tensor and append to a list of tensors
    tensors = torch.jit.annotate(List[Tensor], [])
    for sublist in result:
        tensors.append(torch.tensor(sublist, dtype=torch.int64))

    # Stack the tensors to form a 2D tensor
    final_tensor = torch.stack(tensors)
    return final_tensor.to(device)


@torch.jit.script
def array_transform(x: Tensor, name: str) -> Tensor:
    if name == 'IDENT':
        return x
    elif name == 'RT090':
        return torch.rot90(x, k=1, dims=[0, 1]).to(x.device)
    elif name == 'RT180':
        return torch.rot90(x, k=2, dims=[0, 1]).to(x.device)
    elif name == 'RT270':
        return torch.rot90(x, k=3, dims=[0, 1]).to(x.device)
    elif name == 'FLPLR':
        return torch.flip(x, dims=[1]).to(x.device)
    elif name == 'FLPUD':
        return torch.flip(x, dims=[0]).to(x.device)
    elif name == 'FLPDG':
        temp = torch.rot90(x, k=1, dims=[0, 1]).to(x.device)
        return torch.flip(temp, dims=[0])
    elif name == 'FLPAD':
        temp = torch.rot90(x, k=1, dims=[0, 1]).to(x.device)
        return torch.flip(temp, dims=[1]).to(x.device)
    else:
        raise ValueError("Unknown transform name: " + name)


@torch.jit.script
def color_transform(x: Tensor, name: str) -> Tensor:
    if name == 'CPID':
        mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long, device=x.device)
    elif name == 'CP01':
        mapping = torch.tensor([7, 4, 5, 2, 8, 3, 0, 9, 6, 1], dtype=torch.long, device=x.device)
    elif name == 'CP02':
        mapping = torch.tensor([0, 9, 4, 5, 6, 8, 1, 3, 2, 7], dtype=torch.long, device=x.device)
    elif name == 'CP03':
        mapping = torch.tensor([7, 4, 1, 9, 6, 0, 8, 2, 5, 3], dtype=torch.long, device=x.device)
    elif name == 'CP04':
        mapping = torch.tensor([9, 6, 5, 7, 4, 0, 3, 8, 1, 2], dtype=torch.long, device=x.device)
    elif name == 'CP05':
        mapping = torch.tensor([1, 8, 0, 3, 9, 5, 6, 2, 7, 4], dtype=torch.long, device=x.device)
    elif name == 'CP06':
        mapping = torch.tensor([5, 3, 1, 9, 7, 6, 0, 2, 8, 4], dtype=torch.long, device=x.device)
    elif name == 'CP07':
        mapping = torch.tensor([1, 4, 3, 8, 7, 9, 6, 2, 5, 0], dtype=torch.long, device=x.device)
    elif name == 'CP08':
        mapping = torch.tensor([6, 0, 2, 1, 3, 4, 7, 8, 5, 9], dtype=torch.long, device=x.device)
    elif name == 'CP09':
        mapping = torch.tensor([2, 0, 3, 8, 4, 6, 1, 9, 5, 7], dtype=torch.long, device=x.device)
    else:
        raise ValueError("Unknown color permutation name: " + name)
    
    x_long = x.long()
    return mapping[x_long]


def shuffled_indices(N: int) -> List[int]:
    # Shuffle the list using Fisher-Yates algorithm
    indices = list(range(N))
    for i in range(N - 1, 0, -1):
        # Generate a random index j such that 0 <= j <= i
        j = torch.randint(0, i + 1, (1,)).item()
        
        # Swap product_transforms[i] with product_transforms[j]
        temp = indices[i]
        indices[i] = indices[j]
        indices[j] = temp

    return indices


@torch.jit.script
def augmentations() -> List[Tuple[Tuple[int, str], Tuple[int, str]]]:
    color_transforms = ['CPID', 'CP01', 'CP02', 'CP03', 'CP04', 'CP05', 'CP06', 'CP07', 'CP08', 'CP09']
    array_transforms = ['IDENT', 'RT090', 'RT180', 'RT270', 'FLPLR', 'FLPUD', 'FLPDG', 'FLPAD']
    
    product_transforms = torch.jit.annotate(List[Tuple[Tuple[int, str], Tuple[int, str]]], [])
    for cid, c in enumerate(color_transforms):
        for aid, a in enumerate(array_transforms):
            product_transforms.append(((cid, c), (aid, a)))

    identity_augment = product_transforms[0]
    product_transforms = product_transforms[1:]
    N = len(product_transforms)
    indices = shuffled_indices(N)
    product_transforms = [product_transforms[i] for i in indices]
    return [identity_augment] + product_transforms




@torch.jit.script
def collate_fnc(ex: Example, augmentation: Tuple[Tuple[int, str], Tuple[int, str]], prog_idx: int = 0, device: str = 'cpu') -> Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]:
    x = ex.input

    # Move input to the specified device
    x = x.to(device)

    cpid = augmentation[0][0]
    aid =  augmentation[1][0]

    x = color_transform(x, augmentation[0][1])
    x = array_transform(x, augmentation[1][1])
    inpt_grid, inpt_indices = serialize_array(x)

    inp = MODEL_INPUT(
        is_inverse=torch.tensor([[0]], device=x.device),
        color_permutation=torch.tensor([[cpid]], device=x.device),
        array_transform=torch.tensor([[aid]], device=x.device),
        program=torch.tensor([[prog_idx]], device=x.device),
        grid=inpt_grid.unsqueeze(0),
        grid_indices=inpt_indices.unsqueeze(0)
    )

    y = ex.output
    if y is None:
        return inp, None
    
    y = y.to(device)
    y = color_transform(y, augmentation[0][1])
    y = array_transform(y, augmentation[1][1])
    out_grid, out_indices = serialize_array(y)
    target_grid = torch.cat([out_grid[1:], torch.tensor([0], device=x.device)], dim=0)

    out = MODEL_OUTPUT(
        grid=out_grid.unsqueeze(0),
        grid_indices=out_indices.unsqueeze(0),
        target_grid=target_grid.unsqueeze(0))

    return inp, out


def load_tasks(tasks_json_path: str, solution_path: Optional[str] = None, sort_by_complexity=True) -> List[Task]:
    json_tasks = json.load(open(tasks_json_path, 'r'))
    solutions = json.load(open(solution_path, 'r')) if solution_path is not None else {}

    tasks = []
    for task_id, task in json_tasks.items():
        train_examples = []
        for e in task['train']:
            example = Example(
                input=torch.tensor(e['input']),
                output=torch.tensor(e['output'])
            )
            train_examples.append(example)

        test_examples = []
        for idx, e in enumerate(task['test']):
            output = torch.tensor(solutions[task_id][idx]) if task_id in solutions else None
            example = Example(
                input=torch.tensor(e['input']),
                output=output
            )
            test_examples.append(example)
        
        task = Task(task_id, train_examples, test_examples)
        tasks.append(task)

    if sort_by_complexity:
        tasks = sorted(tasks, key=lambda x: x.complexity())

    return tasks


@torch.jit.script
def split_task(task: Task, device: str = 'cpu'):
    augments: List[Tuple[Tuple[int, str], Tuple[int, str]]] = augmentations()
    train_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []
    eval_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []
    test_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []

    for example in task.train:
        for i, augment in enumerate(augments):
            inp, out = collate_fnc(example, augment, device=device)
            if i == 0:
                eval_data.append((inp, out))
            else:
                train_data.append((inp, out))

    for example in task.test:
        inp, out = collate_fnc(example, augments[0], device=device)
        test_data.append((inp, out))

    # Shuffle the training/eval data
    train_data = [train_data[i] for i in shuffled_indices(len(train_data))]
    eval_data = [eval_data[i] for i in shuffled_indices(len(eval_data))]

    return train_data, eval_data, test_data


@torch.jit.script
def split_task_cross(task: Task, device: str = 'cpu', mode: str = 'Rv1'):
    augments: List[Tuple[Tuple[int, str], Tuple[int, str]]] = augmentations()
    train_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []
    eval_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []
    test_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []

    num_train = len(task.train)

    # Applies the 1 vs rest strategy to create training and evaluation data
    for i in range(num_train):
        prog_idx = i + 1 # 0 is reserved for mean program 
        if mode == '1vR':
            train_examples = [task.train[i]]
            eval_examples = task.train[:i] + task.train[i+1:]
        elif mode == 'Rv1':
            train_examples = task.train[:i] + task.train[i+1:]
            eval_examples = [task.train[i]]
        else:
            raise ValueError("Unknown mode: " + mode)
        
        # Add all augmentations for the training example using the prog_id
        for train_example in train_examples:
            for augment in augments:
                inp, out = collate_fnc(train_example,
                                    augment, 
                                    prog_idx=prog_idx, 
                                    device=device)
                train_data.append((inp, out))

        # All other examples are used for evaluation with the same program id (only identity augment for speed of evaluation)
        for example in eval_examples:
            inp, out = collate_fnc(example, augments[0], prog_idx=prog_idx, device=device)
            eval_data.append((inp, out))

    # Test examples are only used for debugging so keep them as reference with identity augment
    for example in task.test:
        inp, out = collate_fnc(example, augments[0], prog_idx=0, device=device)
        test_data.append((inp, out))

    # Shuffle the training/eval data
    train_data = [train_data[i] for i in shuffled_indices(len(train_data))]
    eval_data = [eval_data[i] for i in shuffled_indices(len(eval_data))]

    return train_data, eval_data, test_data


@torch.jit.script
def loss_fn(logits: Tensor, y: MODEL_OUTPUT):
    y: Optional[Tensor] = y.target_grid if y.target_grid is not None else None
    assert y is not None
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='mean', ignore_index=0)
    return loss


class AdamWModule(nn.Module):
    def __init__(
        self, 
        param: torch.Tensor,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        grad_clip: float = 1.0
    ):
        super(AdamWModule, self).__init__()
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.param = param
        self.grad_clip = grad_clip

        self.register_buffer('m', torch.zeros_like(param))
        self.register_buffer('v', torch.zeros_like(param))
        self.register_buffer('t', torch.tensor(0, dtype=torch.int64))


    @torch.jit.export
    def reset(self) -> None:
        self.m.zero_()
        self.v.zero_()
        self.t.zero_()

    @torch.jit.export
    def zero_grad(self) -> None:
        if self.param.grad is not None and self.param.grad.is_floating_point():
            self.param.grad.zero_()

    @torch.jit.export
    def step(self, lr: float = 1e-2, wd: float = 0.05) -> None:

        grad = self.param.grad

        # Clip gradients
        if grad is not None:
            # Compute the norm of the gradient
            grad_norm = torch.norm(grad, p=2)
            # Compute scaling factor
            scaling_factor = torch.min(torch.tensor(1.0), self.grad_clip / (grad_norm + 1e-6))
            # Scale the gradient
            grad = grad * scaling_factor

        # Increment time step
        self.t = self.t + 1

        # Apply weight decay
        if wd != 0.0:
            grad = grad + wd * self.param

        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad

        # Compute bias-corrected estimates
        beta1_correction = 1 - self.beta1 ** self.t.item()
        beta2_correction = 1 - self.beta2 ** self.t.item()
        m_hat = self.m / beta1_correction
        v_hat = self.v / beta2_correction

        # Compute parameter update
        denom = v_hat.sqrt() + self.eps
        new_param = self.param - lr * m_hat / denom
        self.param.data.copy_(new_param.data)

        # with torch.no_grad():
        #     torch.renorm(self.param, 2, 1, maxnorm=1, out=self.param)
        
    
def format_float(val: float, decimals: int) -> str:
    multiplier = 10 ** decimals
    rounded_val = int(val * multiplier + 0.5)
    s = str(rounded_val)

    # Ensure the string has enough digits
    s = s.rjust(decimals + 1, '0')

    # Insert the decimal point at the correct position
    integer_part = ''
    fractional_part = ''
    n = len(s)
    for i in range(n):
        if i < n - decimals:
            integer_part += s[i]
        else:
            fractional_part += s[i]

    result = integer_part + '.' + fractional_part
    return str(result)

@torch.jit.script
def generate_lr_schedule(
    lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_scale: float = 0.1
) -> List[float]:
    max_lr = lr
    min_lr = max_lr * min_lr_scale
    step_until_decay = total_steps - 1  # Adjusted to ensure correct range

    lr_values: List[float] = []  # Initialize empty list to store learning rates

    for step in range(total_steps):
        if step < warmup_steps:
            # Linear warmup
            lr_value = max_lr * (step + 1) / warmup_steps
        elif step >= step_until_decay:
            # After decay ends, use minimum learning rate
            lr_value = min_lr
        else:
            # Linear decay
            decay_ratio = (step - warmup_steps) / (step_until_decay - warmup_steps)
            decay_ratio = max(0.0, min(decay_ratio, 1.0))  # Clamp between 0 and 1
            lr_value = max_lr - decay_ratio * (max_lr - min_lr)
        lr_values.append(lr_value)

    return lr_values

@torch.jit.script
def train_token_count(task: Task) -> int:
    input_count: int = 0
    output_count: int = 0
    for example in task.train:
        input_count += example.input.numel()
        output_opt: Optional[Tensor] = example.output
        if output_opt is not None:
            output_count += output_opt.numel()
    return input_count + output_count


@torch.jit.script
def get_batch_size(task: Task, batch_token_count: int, min_bs: int, max_bs: int)-> int:
    token_count = train_token_count(task)
    bs = math.ceil(batch_token_count/token_count)
    return max(min_bs, min(max_bs, bs))

