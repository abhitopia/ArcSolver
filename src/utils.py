import torch.nn as nn
import random
import numpy as np
import torch
import hashlib
import logging
import time
from rich.logging import RichHandler



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
    elif isinstance(nested, (list, tuple)):
        # Recursively apply the function to each element in the list or tuple
        # Use the type of the original container to preserve the structure
        return type(nested)(map_to_tensors(item, func) for item in nested)
    else:
        # If the item is not a tensor or a nested structure, return it as is
        return nested


def add_logger(obj, log_level, name, file_path):

    assert isinstance(obj, object), 'obj must be an instance of a class'
    # Create a logger object
    logger = logging.getLogger(name)
    logger.setLevel(log_level)  # Set the minimum logging level

    if not logger.handlers:
        # Create a file handler that logs messages to a file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)  # Set the minimum logging level for the file handler
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)

        # Create a stream handler that logs messages to the console
        # stream_handler = logging.StreamHandler()
        stream_handler = RichHandler(show_time=False, show_path=False, show_level=False, tracebacks_word_wrap=False)
        stream_handler.setLevel(log_level)  # Set the minimum logging level for the stream handler
        stream_format = logging.Formatter('%(message)s')
        stream_handler.setFormatter(stream_format)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    obj.logger = logger
    obj.debug = lambda msg: logger.debug(msg)
    obj.info = lambda msg: logger.info(msg)
    obj.error = lambda msg: logger.error(msg)
    obj.warning = lambda msg: logger.warning(msg)
    obj.info(f'Log File: {file_path}')
    return logger


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