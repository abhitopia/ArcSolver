#%%
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
        self.model.share_memory()
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
    
    def process_example(self, example, iters=None, top_k=5, max_length=1024, beam_search=True, greedy_search=False):

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
#%%

# class InferenceWorker:
#     def __init__(self, checkpoint_path, device):
#         """
#         Initialize the worker with a given model checkpoint and device.
#         This will load the model on the specified device.
#         """
#         self.checkpoint_path = checkpoint_path
#         self.device = device
#         self.model = self.initialize_model()

#     def initialize_model(self):
#         """
#         Initialize the model only once and reuse it for all tasks in this worker.
#         """
#         print(f"Initializing model on {self.device} with checkpoint {self.checkpoint_path}")
#         model = InferenceEngine(self.checkpoint_path, device=self.device)
#         return model

#     def process_task(self, task, iters=4):
#         """
#         Process a single task with the loaded model.
#         """
#         self.model(task, iters=iters)
#         return f"Processed task {task}"

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

    def run_in_parallel(self, tasks, iters):
        """
        Run tasks in parallel using a pool of workers.
        Each worker initializes the model once and processes multiple tasks.
        """

        output = {
            "checkpoint_path": self.checkpoint_path,
            "iters": iters,
            "tasks": []
        }

        if self.num_workers == 1:
            # If only one worker, process tasks sequentially

            # Initialize the worker
            init_worker(self.checkpoint_path, self.device, iters)
            print(f"Started processing {len(tasks)} tasks sequentially")

            for idx, task in tqdm(enumerate(tasks)):
                print(f"Processing task {idx}")
                result = process_task_in_worker(task)
                output["tasks"].append(result)

            print(f"Finished processing {len(tasks)} tasks in sequentially")

            return output

        # Create the multiprocessing pool
        with mp.Pool(processes=self.num_workers, initializer=init_worker,
                     initargs=(self.checkpoint_path, self.device, iters)) as pool:
            
            # # Distribute tasks dynamically to the pool
            # results = pool.map(process_task_in_worker, tasks)

            # Use imap_unordered to get results as soon as each task completes
            results = pool.imap_unordered(process_task_in_worker, tasks)


            
            print(f"Started processing {len(tasks)} tasks in parallel")

            # Progress bar to track task completion
            for result in tqdm(results, total=len(tasks)):
                # Do something with each result
                output["tasks"].append(result)

                

        print(f"Finished processing {len(tasks)} tasks in parallel")
        return output


# Example usage:
# if __name__ == "__main__":
