
from typing import Dict, List, NamedTuple, Optional, Tuple
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

class Solution(NamedTuple):
    task_id: str
    predictions: List[List[Tensor]]
    scores: List[List[float]]

class MODEL_INPUT(NamedTuple):
    color_permutation: torch.Tensor
    array_transform: torch.Tensor
    program: torch.Tensor
    grid: torch.Tensor
    grid_indices: torch.Tensor
    meta: Optional[List[Dict[str, str]]] = None

class MODEL_OUTPUT(NamedTuple):
    grid: torch.Tensor
    grid_indices: torch.Tensor
    target_grid: Optional[torch.Tensor] = None


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
    return torch.tensor(tokenized), torch.tensor(indices)

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
def deserialize_array(token_indices: List[int]) -> Tensor:
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
    return final_tensor


@torch.jit.script
def array_transform(x: Tensor, name: str) -> Tensor:
    if name == 'IDENT':
        return x
    elif name == 'RT090':
        return torch.rot90(x, k=1, dims=[0, 1])
    elif name == 'RT180':
        return torch.rot90(x, k=2, dims=[0, 1])
    elif name == 'RT270':
        return torch.rot90(x, k=3, dims=[0, 1])
    elif name == 'FLPLR':
        return torch.flip(x, dims=[1])
    elif name == 'FLPUD':
        return torch.flip(x, dims=[0])
    elif name == 'FLPDG':
        temp = torch.rot90(x, k=1, dims=[0, 1])
        return torch.flip(temp, dims=[0])
    elif name == 'FLPAD':
        temp = torch.rot90(x, k=1, dims=[0, 1])
        return torch.flip(temp, dims=[1])
    else:
        raise ValueError("Unknown transform name: " + name)


@torch.jit.script
def color_transform(x: Tensor, name: str) -> Tensor:
    if name == 'CPID':
        mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    elif name == 'CP01':
        mapping = torch.tensor([7, 4, 5, 2, 8, 3, 0, 9, 6, 1], dtype=torch.long)
    elif name == 'CP02':
        mapping = torch.tensor([0, 9, 4, 5, 6, 8, 1, 3, 2, 7], dtype=torch.long)
    elif name == 'CP03':
        mapping = torch.tensor([7, 4, 1, 9, 6, 0, 8, 2, 5, 3], dtype=torch.long)
    elif name == 'CP04':
        mapping = torch.tensor([9, 6, 5, 7, 4, 0, 3, 8, 1, 2], dtype=torch.long)
    elif name == 'CP05':
        mapping = torch.tensor([1, 8, 0, 3, 9, 5, 6, 2, 7, 4], dtype=torch.long)
    elif name == 'CP06':
        mapping = torch.tensor([5, 3, 1, 9, 7, 6, 0, 2, 8, 4], dtype=torch.long)
    elif name == 'CP07':
        mapping = torch.tensor([1, 4, 3, 8, 7, 9, 6, 2, 5, 0], dtype=torch.long)
    elif name == 'CP08':
        mapping = torch.tensor([6, 0, 2, 1, 3, 4, 7, 8, 5, 9], dtype=torch.long)
    elif name == 'CP09':
        mapping = torch.tensor([2, 0, 3, 8, 4, 6, 1, 9, 5, 7], dtype=torch.long)
    else:
        raise ValueError("Unknown color permutation name: " + name)
    
    x_long = x.long()
    return mapping[x_long]


@torch.jit.script
def augmentations() -> List[Tuple[Tuple[int, str], Tuple[int, str]]]:
    color_transforms = ['CPID', 'CP01', 'CP02', 'CP03', 'CP04', 'CP05', 'CP06', 'CP07', 'CP08', 'CP09']
    array_transforms = ['IDENT', 'RT090', 'RT180', 'RT270', 'FLPLR', 'FLPUD', 'FLPDG', 'FLPAD']
    
    product_transforms = torch.jit.annotate(List[Tuple[Tuple[int, str], Tuple[int, str]]], [])
    for cid, c in enumerate(color_transforms):
        for aid, a in enumerate(array_transforms):
            product_transforms.append(((cid, c), (aid, a)))

    # Shuffle the list using Fisher-Yates algorithm
    N = len(product_transforms)
    for i in range(N - 1, 0, -1):
        # Generate a random index j such that 0 <= j <= i
        j = torch.randint(0, i + 1, (1,)).item()
        
        # Swap product_transforms[i] with product_transforms[j]
        temp = product_transforms[i]
        product_transforms[i] = product_transforms[j]
        product_transforms[j] = temp
    return product_transforms


@torch.jit.script
def collate_fnc(ex: Example, augmentation: Tuple[Tuple[int, str], Tuple[int, str]]) -> Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]:
    x = ex.input
    cpid = augmentation[0][0]
    aid =  augmentation[1][0]

    x = color_transform(x, augmentation[0][1])
    x = array_transform(x, augmentation[1][1])
    inpt_grid, inpt_indices = serialize_array(x)

    inp = MODEL_INPUT(
        color_permutation=torch.tensor([[cpid]]),
        array_transform=torch.tensor([[aid]]),
        program=torch.tensor([[0]]),
        grid=inpt_grid.unsqueeze(0),
        grid_indices=inpt_indices.unsqueeze(0)
    )

    y = ex.output
    if y is None:
        return inp, None

    y = color_transform(y, augmentation[0][1])
    y = array_transform(y, augmentation[1][1])
    out_grid, out_indices = serialize_array(y)
    target_grid = torch.cat([out_grid[1:], torch.tensor([0])], dim=0)

    out = MODEL_OUTPUT(
        grid=out_grid.unsqueeze(0),
        grid_indices=out_indices.unsqueeze(0),
        target_grid=target_grid.unsqueeze(0))

    return inp, out


def load_tasks(tasks_json_path: str, solution_path: Optional[str] = None) -> List[Task]:
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

    return tasks


@torch.jit.script
def split_task(task: Task):
    augments: List[Tuple[Tuple[int, str], Tuple[int, str]]] = augmentations()
    train_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []
    eval_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []
    test_data: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]] = []

    for example in task.train:
        for i, augment in enumerate(augments):
            inp, out = collate_fnc(example, augment)
            if i == 0:
                eval_data.append((inp, out))
            else:
                train_data.append((inp, out))

    for example in task.test:
        inp, out = collate_fnc(example, augments[0])
        test_data.append((inp, out))

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
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        eps: float = 1e-8
    ):
        super(AdamWModule, self).__init__()
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.param = param

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
    def step(self) -> None:
        grad = self.param.grad

        # Increment time step
        self.t = self.t + 1

        # Apply weight decay
        if self.weight_decay != 0.0:
            grad = grad + self.weight_decay * self.param

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
        new_param = self.param - self.lr * m_hat / denom
        self.param.data.copy_(new_param.data)
        
    
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
    return result