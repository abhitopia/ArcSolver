#%%
import json
from pathlib import Path
import torch.multiprocessing as mp
import torch
from tqdm import tqdm

from .interpreter2 import InterpreterConfig, Interpreter
from .dataset import GridTokenizer, ProgramTokenizer, TaskToExamples
#%%

mp.set_start_method("spawn", force=True)


class InferenceWorker:
    def __init__(self, checkpoint_path: str, device='cpu', iters=1):
        self.device = device
        self.model = self.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        # self.model.share_memory()
        self.prog_tokenizer = self.model.prog_tokenizer
        self.grid_tokenizer = self.model.grid_tokenizer
        self.iters = iters

    @staticmethod
    def load_checkpoint(checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
        grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
        model_config = InterpreterConfig.from_dict(state_dict['model_config'])
        checkpoint_model = Interpreter(model_config, prog_tokenizer, grid_tokenizer)
        checkpoint_model.load_state_dict(state_dict['model_state_dict'])
        return checkpoint_model
    
    def tokenize(self, example):
        (p, inp), out = example
        p = self.prog_tokenizer.encode(p)
        inp = self.grid_tokenizer.encode(inp)
        out = self.grid_tokenizer.encode(out)
        return (p, inp), out

    
    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize() # wait for the GPU to finish work
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.synchronize() # wait for the MPS to finish work
            torch.mps.empty_cache() # clear the MPS cache

    
    def process_example(self, example, iters=None, top_k=5, max_length=1024, beam_search=True, greedy_search=False):
        self.clear_gpu_cache()
        # (p, inp), out = example
        (p, inp), out = self.tokenize(example)

        processed_example = {
            "input": inp,
            "output": out,
            "prediction": {}
        }

        if greedy_search:
            prediction = self.model.greedy_search(p, inp, iters, max_length=max_length, eos_token_id=12)
            processed_example["prediction"]["greedy"] = prediction
        if beam_search:
            prediction = self.model.beam_search(p, inp, iters, top_k=top_k, max_length=max_length, eos_token_id=12)
            processed_example["prediction"]["beam"] = prediction
        return processed_example
    
    def __call__(self, task, iters=None, top_k=5, max_length=1024, beam_search=True, greedy_search=False):

        assert beam_search or greedy_search, "At least one of beam_search or greedy_search must be True"
        result = {
            "task_id": task.id,
            "task_version": task.version,
            "dataset": task.dataset,
            "rank": task.rank,
            "train": [],
            "test": []
        }
        train_examples, test_examples = TaskToExamples(join_version=True)(task)

        iters = iters or self.iters

        for example in train_examples:
            processed_example = self.process_example(example, iters, top_k, max_length, beam_search, greedy_search)
            result["train"].append(processed_example)

        for example in test_examples:
            processed_example = self.process_example(example, iters, top_k, max_length, beam_search, greedy_search)
            result["test"].append(processed_example)


        return result
    
# Global worker for each process
_global_worker = None

def init_worker(checkpoint_path, device, iters):
    """
    Initialize the global worker in each process.
    This ensures that the worker is initialized only once per process.
    """
    global _global_worker
    if _global_worker is None:
        _global_worker = InferenceWorker(checkpoint_path=checkpoint_path,
                                         device=device,
                                         iters=iters)

def process_task_in_worker(task):
    """
    Process the task using the global worker.
    This function is called by each worker in the pool.
    """
    global _global_worker
    return _global_worker(task=task, beam_search=True, greedy_search=True)


class InferenceManager:
    def __init__(self, checkpoint_path, num_workers=4, device='cpu'):
        """
        Manage the parallel execution of tasks using multiple workers.
        """
        self.checkpoint_path = checkpoint_path
        self.num_workers = num_workers
        self.device = device

    def save_results(self, output, output_path):
        """
        Save the output to a file.
        """
        with open(output_path, 'w') as f:
            f.write(json.dumps(output, indent=4))

    def run_in_parallel(self, tasks, iters, output_path, save_interval=10):
        """
        Run tasks in parallel using a pool of workers.
        Each worker initializes the model once and processes multiple tasks.
        """

        # Resume from the file if it exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "checkpoint_path": self.checkpoint_path,
            "iters": iters,
            "tasks": []
        }

        if output_path.exists():
            output = json.load(output_path.open('r'))
            
        
        # Remove already processed tasks
        processed_task_ids = set(task["task_id"] for task in output["tasks"])
        remaining_tasks = [task for task in tasks if task.id not in processed_task_ids]

        print(f"Processing {len(remaining_tasks)} out of {len(tasks)} tasks")
   
        progress_bar = tqdm(total=len(tasks))
        progress_bar.update(len(processed_task_ids))

        if self.num_workers == 1:
            # If only one worker, process tasks sequentially

            # Initialize the worker
            init_worker(self.checkpoint_path, self.device, iters)
            print(f"Started processing {len(tasks)} tasks sequentially")


            for idx, task in enumerate(remaining_tasks):
                result = process_task_in_worker(task)
                output["tasks"].append(result)
                progress_bar.update(1)

                if idx % save_interval == 0 and idx > 0:
                    self.save_results(output, output_path)


            print(f"Finished processing {len(tasks)} tasks in sequentially")
            self.save_results(output, output_path)
            return output

        # Create the multiprocessing pool
        with mp.Pool(processes=self.num_workers, initializer=init_worker,
                     initargs=(self.checkpoint_path, self.device, iters)) as pool:
            
            # # Distribute tasks dynamically to the pool
            # results = pool.map(process_task_in_worker, tasks)

            # Use imap_unordered to get results as soon as each task completes
            results = pool.imap_unordered(process_task_in_worker, remaining_tasks)


            print(f"Started processing {len(tasks)} tasks in parallel")


            # Progress bar to track task completion
            for idx, result in enumerate(results):
                output["tasks"].append(result)
                progress_bar.update(1)

                if idx % save_interval == 0 and idx > 0:
                    self.save_results(output, output_path)

                
        self.save_results(output, output_path)
        print(f"Finished processing {len(tasks)} tasks in parallel")
        return output

