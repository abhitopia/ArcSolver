import json
from typing import List, NamedTuple, Optional
import torch
from torch import Tensor
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import os
from tqdm import tqdm
import argparse

class Example(NamedTuple):
    input: torch.Tensor
    output: Optional[Tensor]

class Task(NamedTuple):
    task_id: str
    train: List[Example]
    test: List[Example]

class TaskSolution(NamedTuple):
    task_id: str
    predictions: List[List[Tensor]]
    scores: List[List[float]]

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

def worker(device, input_queue, output_queue, model_path):
    # Set the device for this process
    device = torch.device(device)

    # Load the model onto the assigned device
    model = torch.jit.load(model_path).to(device)
    # model.eval()  # Set model to evaluation mode

    while True:
        try:
            # Get the next input from the queue
            task = input_queue.get(timeout=5)
            if task is None:   # Exit signal received
                break

            # # Process the input data
            print(f"Processing task {task.task_id} on worker {os.getpid()} on device {device}")
            solution = model(task)
            print(f"Task {solution.task_id} processed by worker {os.getpid()} on device {device}")
            solution = TaskSolution(task_id=solution.task_id, predictions=solution[0], scores=solution[1])

            # Put the result into the output queue
            output_queue.put(solution)

        except Exception as e:
            print(f"Exception in worker {os.getpid()}: {e}")
            break

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process inputs using a TorchScript model with multiprocessing.")

    # Add arguments
    parser.add_argument('--mp', type=str, required=True,
                        help='Path to the TorchScript model file.')
    parser.add_argument('--tp', type=str, required=True,
                        help='Path to the input challenges file')
    parser.add_argument('--sp', type=str, required=False, default=None,
                        help='Path to optional solutions file')
    parser.add_argument('--devs', type=str, nargs='+', default=['cuda'],
                        help='List of devices to use (e.g., cpu, cuda:0, cuda:1). Default uses GPU if available.')
    parser.add_argument('--np', type=int, default=2,
                        help='Number of processes to run per device.')
    parser.add_argument('--op', type=str, default='submission.json',
                        help='Path to save the output results.')
    # You can add more arguments as needed

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # Set the multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    # Determine devices to use
    available_devices = set()
    for device in args.devs:
        if device.startswith('cuda'):
            if torch.cuda.is_available():
                # Check if specified GPU is available
                device_id = device.split(':')[1] if ':' in device else '0'
                if int(device_id) < torch.cuda.device_count():
                    available_devices.append(device)
                else:
                    print(f"CUDA device {device} is not available.")
            else:
                print("CUDA is not available. Switching to CPU.")
                available_devices.add('cpu')
        elif device == 'cpu':
            available_devices.add('cpu')
        else:
            print(f"Unknown device '{device}'. Using CPU instead.")
            available_devices.add('cpu')

    available_devices = list(available_devices)
    if not available_devices:
        print("No valid devices specified. Exiting.")
        return

    print("Using devices:", available_devices)

    num_procs_per_device = args.np
    print(f"# Processes per device: {num_procs_per_device}")

    # Total number of processes
    total_procs = len(available_devices) * num_procs_per_device
    print(f"# Total Processes: {total_procs}")

    # Path to the TorchScript model
    model_path = args.mp

    # Load tasks
    # inputs = load_inputs(args.input_path, args.batch_size)
    tasks = load_tasks(args.tp, args.sp)
    print(f"# Tasks {len(tasks)}")


    # Create the input and output queues
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    # Start the worker processes
    processes = []
    for device in available_devices:
        for _ in range(num_procs_per_device):
            p = mp.Process(target=worker, args=(device, input_queue, output_queue, model_path))
            p.start()
            processes.append(p)

    # Enqueue all inputs
    for task in tasks:
        input_queue.put(task)

    # Send exit signals to the workers
    for _ in processes:
        input_queue.put(None)

    # Collect the outputs with a progress bar
    results = []
    num_inputs = len(tasks)

    print("Processing tasks...")
    with tqdm(total=num_inputs) as pbar:
        for _ in range(num_inputs):
            output_data = output_queue.get()
            results.append(output_data)
            pbar.update(1)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # # Save the results
    # save_outputs(results, args.output_path)

    print("Processing complete. Results saved to:", args.output_path)



# def save_outputs(outputs, output_path):
#     """
#     Save the outputs to the specified path.

#     Args:
#         outputs (list): List of output tensors.
#         output_path (str): Path to save the outputs.
#     """
#     # Example: Save outputs as a single file
#     torch.save(outputs, output_path)

if __name__ == '__main__':
    main()
