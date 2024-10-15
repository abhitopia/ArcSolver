from copy import deepcopy
from pathlib import Path
import subprocess
import torch.nn as nn
import random
import numpy as np
import torch
import hashlib
import logging
import time
from rich.logging import RichHandler
from unique_names_generator import get_random_name


def find_factors(number):
    factors = []
    for i in range(1, int(number ** 0.5) + 1):
        if number % i == 0:
            factors.append(i)
            if i != number // i:  # Check for square root to avoid duplicate factors
                factors.append(number // i)
    return sorted(factors)

def get_possible_n_heads(prog_dim):
    model_dim = prog_dim * 4
    possible_n_heads = find_factors(model_dim)
    possible_n_heads = [n_heads for n_heads in possible_n_heads if (model_dim/n_heads) >= 4 and (model_dim/n_heads) < model_dim and (model_dim/n_heads) % 2 == 0 and n_heads >= 4 ]

    if prog_dim > 8:
        possible_n_heads = [n_heads for n_heads in possible_n_heads if n_heads >= 8]
    return possible_n_heads

def generate_random_sweep_config(sweep_dict):
    random_config = {}
    for key, value in sweep_dict.items():
        if callable(value):
            random_config[key] = value()
        elif isinstance(value, list):
            random_config[key] = random.choice(value)
        else:
            random_config[key] = value
    return random_config

def construct_sweep_config(sweep_dict, experiment_name, prog_dim, **kwargs):
    sweep_dict = deepcopy(sweep_dict)
    sweep_dict["experiment"] = experiment_name
    sweep_dict["run"] = get_random_name(separator="-")
    sweep_dict["prog_dim"] = prog_dim
    sweep_dict["heads"] = get_possible_n_heads(prog_dim)
    for key, value in kwargs.items():
        sweep_dict[key] = value
    return sweep_dict

def generate_loguniform_numbers(a, b, n):
    log_a = np.log(a)
    log_b = np.log(b)
    uniform_samples = np.random.uniform(log_a, log_b, n)
    loguniform_samples = np.exp(uniform_samples)
    return loguniform_samples.tolist()


def migrate_hparam_dict(hparam_dict):
    """For backward compatibility with old hyperparameter keys.
    """
    new_keys = {
        'optim.lr_schedule': 'noam'
    }

    migration_dict = {
        'optim.model_lr': 'optim.lr_model',
        'optim.model_wd': 'optim.wd_model',
        'optim.prog_lr': 'optim.lr_prog',
        'optim.prog_wd': 'optim.wd_prog',
    }

    for key, val in new_keys.items():
        if key not in hparam_dict:
            hparam_dict[key] = val
    
    for old_key, new_key in migration_dict.items():
        if old_key in hparam_dict:
            hparam_dict[new_key] = hparam_dict[old_key]
            del hparam_dict[old_key]

    return hparam_dict


def get_diff_dict(dict_src, dict_trg):
    diff_dict = {}
    for key in dict_trg:
        if dict_src[key] != dict_trg[key]:
            diff_dict[key] = f"{key}: {dict_src[key]} -> {dict_trg[key]}"

    return diff_dict

def get_git_commit_hash(n):
    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return commit_hash[:n]
    except subprocess.CalledProcessError:
        # Handle the case where the git command fails (e.g., not a git repository)
        print("Failed to retrieve git commit hash")
        return None

def nearest_greater_power_of_2(n):
    if n <= 0:
        raise ValueError("Number must be positive")
    # Find the bit length of n and compute 2 raised to this bit length
    power = 2 ** (n - 1).bit_length()
    return power


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self  # This allows usage of the instance within the with block

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.interval_ms = (self.end - self.start) * 1000


def map_to_tensors(nested, func):
    """
    Apply a function to every tensor in a nested structure.

    Args:
    - nested: The nested structure containing tensors.
    - func: A function that takes a tensor as input and returns a tensor.

    Returns:
    - The nested structure with the function applied to every tensor.
    """
    if isinstance(nested, torch.Tensor):
        # Apply the function to the tensor
        return func(nested)
    elif isinstance(nested, dict):
        # Recursively apply the function to each value in the dictionary
        return {key: map_to_tensors(value, func) for key, value in nested.items()}
    elif isinstance(nested, tuple) and hasattr(nested, '_fields'):
        # It's a NamedTuple
        # Map to each field and reconstruct the NamedTuple
        return type(nested)(*(map_to_tensors(getattr(nested, field), func) for field in nested._fields))
    elif isinstance(nested, (list, tuple)):
        # Recursively apply the function to each element in the list or tuple
        # Use the type of the original container to preserve the structure
        return type(nested)(map_to_tensors(item, func) for item in nested)
    else:
        # If the item is not a tensor or a nested structure, return it as is
        return nested
    
def get_logger(name: str = None):
    logger = logging.getLogger("ArcSolver" if name is None else name)

    # You can set change the level to logging.DEBUG later
    # Even after the initialisation
    logger.setLevel(logging.INFO)  # Set the minimum logging level
    # The logger acts as a gateway to the logger handlers with
    # handlers processing anything at and above their set levels.

    if len(logger.handlers) == 0:
        # Create a stream handler that logs messages to the console
        # stream_handler = logging.StreamHandler()
        stream_handler = RichHandler(show_time=False, show_path=False, show_level=False, tracebacks_word_wrap=False)
        stream_handler.setLevel(logging.DEBUG)  # Set the minimum logging level for the stream handler
        stream_format = logging.Formatter('%(message)s')
        stream_handler.setFormatter(stream_format)
        logger.addHandler(stream_handler)

    return logger

def add_logging_funcs(obj, logger=None):
    logger = logger if logger is not None else get_logger()
    obj.debug = lambda msg: logger.debug(msg)
    obj.info = lambda msg: logger.info(msg)
    obj.error = lambda msg: logger.error(msg)
    obj.warning = lambda msg: logger.warning(msg)

 
def add_logfile_handler(file_path, logger=None):
    if logger is None:
        logger = logging.getLogger()

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.DEBUG)  # Set the minimum logging level for the file handler
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {file_path}")



def task_stats(tasks):
    # For each dataset, find the number of tasks, number of training examples, number of test examples
    # Keep track of total number of tasks, total number of training examples, total number of test examples
    stats = {}

    total_num_tasks = 0
    total_num_train_examples = 0
    total_num_test_examples = 0

    for task in tasks:
        dataset = task.dataset
        if dataset not in stats:
            stats[dataset] = {'num_tasks': 0, 'num_train_examples': 0, 'num_test_examples': 0}
        stats[dataset]['num_tasks'] += 1
        stats[dataset]['num_train_examples'] += len(task.train)
        stats[dataset]['num_test_examples'] += len(task.test)
        total_num_tasks += 1
        total_num_train_examples += len(task.train)
        total_num_test_examples += len(task.test)

    stats['TOTAL'] = {'num_tasks': total_num_tasks, 'num_train_examples': total_num_train_examples, 'num_test_examples': total_num_test_examples}

    ## Nicely print the stats 
    for dataset, stat in stats.items():
        print(f'{dataset}: {stat}')

def hash_string(my_string: str, len: int=8) -> str:
    hash_object = hashlib.sha256(my_string.encode())  # Encode the string to bytes
    hash_code = hash_object.hexdigest()  # Get the hexadecimal digest of the hash
    return hash_code[:len]  # Return the first `len` characters of the hash


def count_trainable_parameters(model: nn.Module):
    unique_parameters = set()
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:
            # Check if the parameter has not been counted yet based on its memory location
            if param.data_ptr() not in unique_parameters:
                unique_parameters.add(param.data_ptr())
                total_params += param.numel()
    return total_params


def seed_everything(seed=1337):
    random.seed(seed)            # Python's built-in random module
    np.random.seed(seed)         # NumPy's random module
    torch.manual_seed(seed)      # PyTorch's random number generator for CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # PyTorch's random number generator for CUDA
        torch.cuda.manual_seed_all(seed)     # for multi-GPU setups
        torch.backends.cudnn.deterministic = True  # To increase reproducibility on GPU
        torch.backends.cudnn.benchmark = False


def flatten_dict(config_dict: dict) -> dict:
        # Flatten the dictionary and convert lists to strings
    flat_config_dict = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # Flatten nested dictionaries by prefixing keys with the parent key
            flat_dict = flatten_dict(value)
            for subkey, subvalue in flat_dict.items():
                flat_config_dict[f"{key}.{subkey}"] = subvalue
        elif isinstance(value, list):
            # Convert lists to string
            flat_config_dict[key] = ', '.join(value)
        else:
            flat_config_dict[key] = value
    return flat_config_dict

def int64_tensor_memory(tensors):
    # Calculate the total number of elements
    total_elements = sum([t.numel() for t in tensors])

    # Size of each element in bytes (int64 has 8 bytes)
    element_size_bytes = 8

    # Total memory consumption in bytes
    total_memory_bytes = total_elements * element_size_bytes

    # Optionally, convert bytes to kilobytes (KB) or megabytes (MB)
    total_memory_kb = total_memory_bytes / 1024
    total_memory_mb = total_memory_kb / 1024

    print(f"Total memory consumption: {total_memory_bytes} bytes")
    print(f"Total memory consumption: {total_memory_kb} KB")
    print(f"Total memory consumption: {total_memory_mb} MB")