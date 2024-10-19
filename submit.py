import argparse
import json
import os
from pathlib import Path
import random
import threading
import time
from typing import Dict, List, NamedTuple, Optional, Union, get_type_hints
import torch
import torch.jit
from queue import Empty
from tqdm import tqdm
import warnings
from torch import Tensor
import hashlib
import torch.multiprocessing as mp
from typing import get_origin, get_args, get_type_hints, Union, Optional



# Suppress the specific RNN UserWarning
warnings.filterwarnings(
    "ignore",
    message=r".*RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning
)

warnings.filterwarnings("ignore", category=UserWarning, message=r".*resource_tracker:.*")


def deterministic_shuffle_local(N, seed=42):
    numbers = list(range(N))
    rng = random.Random(seed)
    rng.shuffle(numbers)
    return numbers


def is_queue_closed(queue):
    return queue._closed

def drain_queue(queue):
    """Drains the queue by retrieving and discarding all items."""

    if is_queue_closed(queue):
        print("Main: Queue is already closed.")
        return

    print("Main: Draining queue...")
    try:
        while True:
            queue.get_nowait()  # Remove items without waiting
    except Empty:
        pass  # When the queue is empty, stop draining



def checksum_json_file(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    # Serialize data with sorted keys and compact separators
    serialized_data = json.dumps(data, sort_keys=True, separators=(',', ':'))
    checksum = hashlib.md5(serialized_data.encode('utf-8')).hexdigest()
    return checksum


class Example(NamedTuple):
    input: torch.Tensor
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

    def to_dict(self):
        result = []
        for pred in self.predictions:
            pred1 = pred[0].tolist()
            pred2 = pred[1].tolist() if len(pred) > 0 else pred1
            result.append({'attempt_1': pred1, 'attempt_2': pred2})
        return result

class ModelParams(NamedTuple):
    thinking: int = 250
    btc: int = 8000
    min_bs: int = 4
    max_bs: int = 16
    patience: int = 50
    lr: float = 0.01
    wd: float = 0.0
    wu: int = 1
    lrs: float = 0.5
    seed: int = 60065
    metric: str = 'L'
    strategy: str = 'Rv1'
    zero_init: bool = False # Whether to zero initialize the program embedding solution
    mode: str = '60065'
    predict: bool = True # Whether to return the solution or not, used in evaluation mode to save time
    return_logs: bool = False
    top_k: Optional[int] = 3
    num_beams: int = 9


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
        print("Sorting tasks by complexity")
        tasks = sorted(tasks, key=lambda x: x.complexity())
    return tasks

def create_dummy_submission(tasks: List[Task]):
    submission = {}
    for task in tasks:
        submission[task.task_id] = []
        for _ in task.test:
            submission[task.task_id].append({
                'attempt_1': [[6, 0, 0, 6, 5], [0, 4, 5, 5]],
                'attempt_2': [[6, 0, 0, 6, 5], [0, 4, 5, 5]]
            })

    return submission


class Worker(mp.Process):
    def __init__(self, device_id, worker_id, model_path, input_queue, output_queue, model_params: ModelParams, start_time, time_limit_seconds):
        super(Worker, self).__init__()
        self.device_id = device_id
        self.worker_id = worker_id
        self.model_path = model_path
        self.model_params = model_params
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.start_time = start_time
        self.time_limit_seconds = time_limit_seconds
        self.model = None  # Model will be loaded once in the run method

    def load_model(self):
        """Load the model onto the correct device."""
        device = torch.device(f'cuda:{self.device_id}' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(self.model_path).to(device)
        print(f"Worker {self.worker_id}: Model loaded on {device}")

    def run(self):

     # Load the model once when the process starts
        self.load_model()
        while True:
            try:
                try:
                    # Get the next task from the input queue with timeout
                    task = self.input_queue.get(timeout=1)
                except Empty:
                    continue

                # If None is received, this is the signal to terminate
                if task is None:
                    self.input_queue.task_done()  # This call is correct for 'None' signal.
                    print(f"Worker {self.worker_id} exiting.")

                    time.sleep(5) # This is super important for processing without error. No one knows why
                    break

                # Log task processing
                print(f"Worker {self.worker_id} ({task.task_id}): Started")

                try:

                    if torch.cuda.is_available():
                        torch.cuda.synchronize() # wait for the GPU to finish work
                        torch.cuda.empty_cache()

                    # Run the task
                    solution = self.model.forward(task, self.model_params)
                    result = TaskSolution(solution[0], solution[1], solution[2], solution[3])

                    # Task processing is done, mark the task as complete
                    self.input_queue.task_done()
                    # Put the result (solution or exception) into the output queue
                    self.output_queue.put(result)
                    print(f"Worker {self.worker_id}: Successfully put the result in the output queue")

                except Exception as task_error:
                    # Handle exceptions during task processing
                    print(f"Worker {self.worker_id}: Error during task {task.task_id}: {task_error}")
                    self.output_queue.put(task_error)
                    self.input_queue.task_done()  # Mark the task as done even if there was an error.

            except Exception as general_error:
                print(f"Worker {self.worker_id}: General error occurred: {general_error}")
                self.output_queue.put(general_error)  # Send the exception back to the main process
                # Do NOT call task_done() here, as there might not have been a task retrieved.


class SubmissionManager:
    def __init__(self, model_path, num_devices, threads_per_device, model_params: ModelParams, submission_path, time_limit_seconds):
        self.model_path = model_path
        self.num_devices = num_devices
        self.threads_per_device = threads_per_device
        self.model_params = model_params
        self.submission_path = submission_path
        self.time_limit_seconds = time_limit_seconds  # Convert time limit to seconds

        assert num_devices > 0, "At least one device must be available"
        assert threads_per_device > 0, "At least one thread per device is required"

        self.input_queue = mp.JoinableQueue()  # Multiprocessing-compatible queue
        self.output_queue = mp.Queue()
        self.workers = []
        self.results = None
        self.solutions = {}
        self.num_inputs = 0
        self.start_time = time.time()  # Track the start time

        submission_file_name = Path(submission_path).name
        self.solutions_path = Path(submission_path).parent / f"{submission_file_name}_solutions.pkl"
        self.save_solutions = model_params.return_logs
    
    def start_workers(self):
        """Starts the worker processes."""
        for device_id in range(self.num_devices):
            for n in range(self.threads_per_device):
                worker_id = device_id * self.threads_per_device + n
                worker = Worker(
                    device_id, worker_id, self.model_path,
                    self.input_queue, self.output_queue,
                    self.model_params, self.start_time,
                    self.time_limit_seconds
                )
                worker.start()
                self.workers.append(worker)

    def add_inputs(self, tasks):
        """Add tasks to the input queue."""
        self.num_inputs = len(tasks)
        self.results = create_dummy_submission(tasks)
        for task in tasks:
            self.input_queue.put(task)
        # Add termination signals (None) to stop workers
        for _ in self.workers:
            self.input_queue.put(None)

    def terminate_workers(self):
        """Forcefully terminate all running workers."""
        print("Main: Terminating workers...")
        for worker in self.workers:
            if worker.is_alive():
                print(f"Terminating worker {worker.pid}")
                worker.terminate()
        # Now, wait for workers to terminate, with a timeout
        for worker in self.workers:
            if worker.is_alive():
                print(f"Joining worker {worker.pid} with timeout")
                worker.join(timeout=5)
                if worker.is_alive():
                    print(f"Worker {worker.pid} did not terminate in time")

        self.close_queues()

    def all_workers_finished(self):
        return all(not worker.is_alive() for worker in self.workers)
    

    def close_queues(self):
        try:
            drain_queue(self.input_queue)
            drain_queue(self.output_queue)
            self.output_queue.close()
            self.input_queue.close()
            self.output_queue.join_thread()
            self.input_queue.join_thread()
        except Exception as e:
            print(f"Exception while closing input/output queues: {e}")

    def process_inputs(self):
        num_processed = 0
        with tqdm(total=self.num_inputs) as pbar:
            while num_processed < self.num_inputs:

                elapsed_time = time.time() - self.start_time
                if elapsed_time > self.time_limit_seconds:
                    print(f"Main: Time limit reached ({elapsed_time:.2f} seconds). Terminating workers.")
                    self.terminate_workers()
                    break

                try:
                    # Get a result from the output queue with timeout
                    solution = self.output_queue.get(timeout=1)
                    print(f"Main: Received result for {solution.task_id}")
                    if isinstance(solution, Exception):
                        print(f"Main: Error encountered in task: {solution}")
                    else:
                        # Process the result
                        self.results[solution.task_id] = solution.to_dict()
                        self.solutions[solution.task_id] = solution
                        json.dump(self.results, open(self.submission_path, 'w'), indent=2)

                        if self.save_solutions:
                            torch.save(self.solutions, self.solutions_path)
        
                        print(f"Main: Saved Solution: Task {solution.task_id}")
                    num_processed += 1
                    pbar.update(1)

                except Empty:
                    # If no result is available, continue waiting
                    if self.all_workers_finished():  # Check if all workers are done
                        print("Main: All workers finished, no more results.")
                        break  # Exit the loop if all workers are done


    def get_results(self):
        """Returns the final results."""
        return self.results
    

def parse_args_from_namedtuple(namedtuple_class, parser):    
    # Get type hints and default values
    type_hints = get_type_hints(namedtuple_class)
    defaults = namedtuple_class._field_defaults

    for field, field_type in type_hints.items():
        default_value = defaults.get(field)
        arg_type = None

        # Determine the base type, handling Optional and Union types
        origin_type = get_origin(field_type)
        if origin_type is Union:
            # Handle Optional types (Union[..., NoneType])
            args = get_args(field_type)
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                arg_type = non_none_types[0]
            else:
                # If multiple non-None types, default to str or handle accordingly
                arg_type = str
        else:
            arg_type = field_type

        # Map Python types to argparse argument types
        if arg_type == int:
            parser.add_argument(f"--{field}", type=int, default=default_value, help=f"{field} (default: {default_value})")
        elif arg_type == float:
            parser.add_argument(f"--{field}", type=float, default=default_value, help=f"{field} (default: {default_value})")
        elif arg_type == str:
            parser.add_argument(f"--{field}", type=str, default=default_value, help=f"{field} (default: {default_value})")
        elif arg_type == bool:
            # For booleans, use appropriate action
            if default_value is True:
                parser.add_argument(f"--{field}", action='store_false', help=f"{field} (default: {default_value})")
            else:
                parser.add_argument(f"--{field}", action='store_true', help=f"{field} (default: {default_value})")
        else:
            # For other or unknown types, default to str
            parser.add_argument(f"--{field}", type=str, default=default_value, help=f"{field} (default: {default_value})")

    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process ")
    # Add arguments
    parser.add_argument('--mp', type=str, required=True,
                        help='Path to the model file.')
    parser.add_argument('--tp', type=str, required=False, default='/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json',
                        help='Path to the input challenges file')
    parser.add_argument('--sp', type=str, required=False, default=None,
                        help='Path to optional solutions file')
    parser.add_argument('--np', type=int, default=5, required=False,
                        help='Number of processes to run per device.')
    parser.add_argument('--op', type=str, default='/kaggle/working/submission.json',
                        help='Path to save the output results.')
    parser.add_argument('--tl', type=int, default=int(11.5*3600), help=f'Time limit in seconds for processing tasks. {11.5} hours is the default limit.')

    parser.add_argument('--nt', type=int, default=-1, help='Number of randomly chosen tasks')
    
    # Model Arguments
    parser = parse_args_from_namedtuple(ModelParams, parser)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    total_time_limit = args.tl  # Original time limit in seconds
    grace_period = 60  # 60-second grace period
    hard_exit_time = total_time_limit + grace_period

    print(f"Total Time Limit: {total_time_limit} seconds")
    print(f"Grace Period: {grace_period} seconds")
    print(f"Hard Exit Time: {hard_exit_time} seconds")



    # Filter out extra arguments that aren't in SolverParams
    model_params_dict = {key: value for key, value in vars(args).items() if key in ModelParams._fields}

    # Create the SolverParams instance with only the required fields
    model_params = ModelParams(**model_params_dict)

    print("Model Parameters:")
    for key, value in model_params._asdict().items():
        print(f"\t{key}: {value}")

    model_path = args.mp
    tasks_path = args.tp
    solutions_path = args.sp
    output_path = args.op

    assert Path(model_path).exists(), f"Model file not found: {model_path}"
    assert Path(tasks_path).exists(), f"Tasks file not found: {tasks_path}"
    assert Path(output_path).parent.exists(), f"Output path not found: {output_path}"

    if solutions_path is not None:
        assert Path(solutions_path).exists(), f"Solutions file not found: {solutions_path}"
    
    tasks = load_tasks(tasks_json_path=tasks_path, solution_path=solutions_path, sort_by_complexity=True)

    if args.nt > 0:
        chosen_indices = deterministic_shuffle_local(len(tasks), seed=42)[:args.nt]
        tasks = [tasks[i] for i in chosen_indices]
        print(f"Randomly selected {args.nt} tasks")
    else:
        print(f"Number of Tasks: {len(tasks)}")

    checksum = checksum_json_file(tasks_path)
    if checksum == 'f346f614a275b133f1a88044ae38468d':
        print('Detected Dummy Test Data')
        print("Creating Dummy Submission")
        output_path = args.op
        submission = create_dummy_submission(tasks)
        json.dump(submission, open(output_path, 'w'), indent=2)
        print(f"Saved Dummy Submission: {output_path}")
        return

    # Number of devices (GPUs) and processes per device
    D = torch.cuda.device_count() 
    N = args.np   # Number of model instances per device

    devices = [f'cuda:{i}' for i in range(D)] if D > 0 else ['cpu']
    D = 1 if D == 0 else D  # Use CPU if no GPUs are available
    
    print(f"Using devices: {devices}")
    print(f'Number of Cuda devices: {D}')
    print(f'Number of threads per device: {N}')

    # Initialize the processing manager with a time limit
    manager = SubmissionManager(model_path=model_path, 
                                num_devices=D, 
                                threads_per_device=N, 
                                model_params=model_params,
                                submission_path=output_path, 
                                time_limit_seconds=args.tl)  # Set time limit (e.g., 10 hours)


    def force_exit():
        print("Main: Force exiting due to time limit.")
        manager.terminate_workers()  # Terminate all worker processes
        os._exit(0)

    # Start a timer to force exit after hard_exit_time
    timer = threading.Timer(hard_exit_time, force_exit)
    timer.start()

    try:
        # Start worker processes
        manager.start_workers()

        # Add inputs to the queue
        manager.add_inputs(tasks)

        # Process inputs and collect outputs, with time limit checking
        manager.process_inputs()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Terminating workers.")
        manager.terminate_workers()
        raise # Re-raise the KeyboardInterrupt
    finally:
        manager.terminate_workers()
        timer.cancel()  # Cancel the timer if we finished before time limit


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # Use spawn method for multiprocessing with CUDA
    main()
