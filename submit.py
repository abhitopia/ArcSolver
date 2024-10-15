import argparse
import json
from pathlib import Path
from typing import List, NamedTuple, Optional, get_type_hints
import torch
import torch.jit
import threading
from queue import Queue
from tqdm import tqdm
import warnings
from torch import Tensor
import hashlib

# Suppress the specific RNN UserWarning
warnings.filterwarnings(
    "ignore",
    message=r".*RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning
)


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

class TaskSolution(NamedTuple):
    task_id: str
    predictions: List[List[Tensor]]
    scores: List[List[float]]

    def to_dict(self):
        result = []
        for pred in self.predictions:
            pred1 = pred[0].tolist()
            pred2 = pred[1].tolist()
            result.append({'attempt_1': pred1, 'attempt_2': pred2})
        return result

class ModelParams(NamedTuple):
    thinking: int = 500
    bs: int = 25
    patience: int = 30
    lr: float = 0.005
    wd: float = 0.05
    wu: int = 10
    lrs: float = 0.1
    seed: int = 60065
    mode: str = '60065'
    confidence: float = 0.0001
    metric: str = 'L'


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

# Worker thread class
class Worker(threading.Thread):
    def __init__(self, device_id, worker_id, model_path, input_queue, output_queue, model_params: ModelParams):
        threading.Thread.__init__(self)
        self.device_id = device_id
        self.worker_id = worker_id
        self.model_params = model_params
        self.input_queue = input_queue
        self.output_queue = output_queue
        # Load the model instance on the assigned device
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path).to(device)
        # self.model.eval()
        self.device = device
        self.daemon = True  # Ensures threads exit when the main thread exits

    def run(self):

        while True:
            try:
                # Free up GPU cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Get the next task from the input queue
                task = self.input_queue.get()

                # If None is received, this is the signal to terminate
                if task is None:
                    print(f"Worker {self.worker_id} exiting.")
                    break

                # Log task processing
                print(f"Worker {self.worker_id} processing task {task.task_id}")

                # Process the input asynchronously
                try:
                    # Use torch.jit.fork to process the input
                    fut = torch.jit.fork(self.model.forward, task, self.model_params)
                    
                    # Wait for the result without blocking the thread
                    solution = torch.jit.wait(fut)

                    # Prepare the task solution
                    solution = TaskSolution(solution[0], solution[1], solution[2])

                    # Log task completion
                    print(f"Worker {self.worker_id} completed task {solution.task_id}")

                    # Put the result into the output queue
                    self.output_queue.put(solution)

                except Exception as e:
                    print(f"Worker {self.worker_id}: Error during task processing for task {task.task_id}: {e}")
                    # Log failure if needed and optionally add retry logic

            except Exception as e:
                print(f"Worker {self.worker_id}: General error occurred: {e}")

            finally:
                # Ensure that the task is marked as done
                self.input_queue.task_done()
# Processing Manager class
class SubmissionManager:
    def __init__(self, model_path, num_devices, threads_per_device, model_params: ModelParams, submission_path):
        self.model_path = model_path
        self.num_devices = num_devices
        self.threads_per_device = threads_per_device
        self.model_params = model_params
        self.submission_path = submission_path

        assert num_devices > 0, "At least one device must be available"
        assert threads_per_device > 0, "At least one thread per device is required"
        
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.workers = []
        self.results = None
        self.num_inputs = 0

    def start_workers(self):
        for device_id in range(self.num_devices):
            for n in range(self.threads_per_device):
                worker_id = device_id * self.threads_per_device + n
                worker = Worker(device_id, worker_id, self.model_path, self.input_queue, self.output_queue, self.model_params)
                worker.start()
                self.workers.append(worker)

    def add_inputs(self, tasks):
        self.num_inputs = len(tasks)
        self.results = create_dummy_submission(tasks)
        # Add inputs to the input queue
        for task in tasks:
            self.input_queue.put(task)
        # Add termination signals to the input queue
        for _ in self.workers:
            self.input_queue.put(None)

    def process_inputs(self):
        num_processed = 0
        with tqdm(total=self.num_inputs) as pbar:
            while num_processed < self.num_inputs:
                solution = self.output_queue.get()
                self.results[solution.task_id] = solution.to_dict()
                json.dump(self.results, open(self.submission_path, 'w'), indent=2)
                print(f"Saved Solution: Task {solution.task_id}")
                num_processed += 1
                pbar.update(1)

        # Wait for all tasks to be processed
        self.input_queue.join()
        # Wait for all worker threads to finish
        for worker in self.workers:
            worker.join()

    def get_results(self):
        return self.results
    

def parse_args_from_namedtuple(namedtuple_class, parser):    
    # Get type hints and default values
    type_hints = get_type_hints(namedtuple_class)
    defaults = namedtuple_class._field_defaults

    for field, field_type in type_hints.items():
        default_value = defaults.get(field)
        
        # Map Python types to argparse argument types
        if field_type == int:
            parser.add_argument(f"--{field}", type=int, default=default_value, help=f"{field} (default: {default_value})")
        elif field_type == float:
            parser.add_argument(f"--{field}", type=float, default=default_value, help=f"{field} (default: {default_value})")
        elif field_type == str:
            parser.add_argument(f"--{field}", type=str, default=default_value, help=f"{field} (default: {default_value})")
        else:
            parser.add_argument(f"--{field}", default=default_value, help=f"{field} (default: {default_value})")

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
    
    # Model Arguments
    parser = parse_args_from_namedtuple(ModelParams, parser)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    # Filter out extra arguments that aren't in SolverParams
    model_params_dict = {key: value for key, value in vars(args).items() if key in ModelParams._fields}

    # Create the SolverParams instance with only the required fields
    model_params = ModelParams(**model_params_dict)
    model_path = args.mp
    tasks_path = args.tp
    solutions_path = args.sp

    assert Path(model_path).exists(), f"Model file not found: {model_path}"
    assert Path(tasks_path).exists(), f"Tasks file not found: {tasks_path}"


    if solutions_path is not None:
        assert Path(solutions_path).exists(), f"Solutions file not found: {solutions_path}"
    
    tasks = load_tasks(tasks_json_path=tasks_path, solution_path=solutions_path)
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

    # Initialize the processing manager
    manager = SubmissionManager(model_path=model_path, 
                                num_devices=D, 
                                threads_per_device=N, 
                                model_params=model_params,
                                submission_path=args.op)    

    # Start worker threads
    manager.start_workers()

    # Add inputs to the queue
    manager.add_inputs(tasks)

    # Process inputs and collect outputs
    manager.process_inputs()

if __name__ == '__main__':
    main()
