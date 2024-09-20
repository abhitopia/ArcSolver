#%%
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

    
sys.path.insert(0, "/teamspace/studios/this_studio/ArcSolveR")
     

#%%
import numpy as np
import json
import torch
import random

from src.dataset import ArcExamplesDataset, GridTokenizer, ProgramTokenizer
from src.interpreter import Interpreter, InterpreterConfig
from src.lazy_adamw import LazyAdamW


def load_checkpoint(checkpoint_path: str, ref=False):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
    grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
    model_config = InterpreterConfig.from_dict(state_dict['model_config'])
    checkpoint_model = Interpreter(model_config, prog_tokenizer, grid_tokenizer)
    checkpoint_model.load_state_dict(state_dict['model_state_dict'])
    return checkpoint_model


def init_device(_device):
    if _device is None:
        # return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(_device, str):
        return torch.device(_device)
    elif isinstance(_device, torch.device):
        return _device


class ArcSolver:
    def __init__(self, checkpoint_path, device=None) -> None:
        self._checkpoint_path = checkpoint_path
        self._model = None
        self._device = init_device(device)
        self._load_solver()
        self._tokenizer = self._model.grid_tokenizer
        self.reset()

    def _load_solver(self):
        model = load_checkpoint(self._checkpoint_path)
        model_dict = model.config.to_dict()
        model_dict['prog_vocab_size'] = 1
        solver_config = InterpreterConfig.from_dict(model_dict)
        solver = Interpreter(solver_config, prog_tokenizer=None, grid_tokenizer=model.grid_tokenizer)
        model_state_dict = model.state_dict()
        solver_state_dict = {k: v for k, v in model_state_dict.items() if 'pte.weight' not in k}
        solver.load_state_dict(solver_state_dict, strict=False)
        self._model = solver
        self._model.to(self._device)


    def reset(self):

        if torch.cuda.is_available():
            torch.cuda.synchronize() # wait for the GPU to finish work
            torch.cuda.empty_cache()

        self._model.pte.weight.data.fill_(0)
        for param in self._model.parameters():
            param.requires_grad = False
            
        self._model.pte.weight.requires_grad = True

    def tokenize(self, example):

        def serialize_array(array: np.ndarray) -> str:
            list_of_lists = array.tolist()
            array_str = json.dumps(list_of_lists)
            array_str = array_str.replace('\n', '').replace(',','')
            return array_str.replace('[[', '[[ ').replace(']]', ' ]]').replace('] [', ' ],[ ')

        inp_arr, out_arr = example
        p = [0]
        inp = self._tokenizer.encode(serialize_array(inp_arr))
        out = self._tokenizer.encode(serialize_array(out_arr))
        return (p, inp), out
        

    def example_to_batch(self, examples, combined=False):

        tokenized_examples = [self.tokenize(example) for example in examples]
        
        if combined:
            batch = ArcExamplesDataset.collate_fn(batch=tokenized_examples,
                                        pad_idx=self._tokenizer.PAD_IDX,
                                        seq_length=None, device=self._device)
            batch = [batch]
        else:
            batch = []
            for example in tokenized_examples:
                batch.append(ArcExamplesDataset.collate_fn(batch=[example],
                                        pad_idx=self._tokenizer.PAD_IDX,
                                        seq_length=None, device=self._device))
                
        return batch


    def _evaluate(self, batch):
        (p, i, l), (y, y_l) = batch
        logits, _, _ = self._model(p, i, l, 10)

        logits, _, _  = self._model(p, i, l, 8)
        loss = self._model.loss_fn(logits, y)

        ## Sample Accuracy
        output_mask = y != GridTokenizer().PAD_IDX
    
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        mask_correct_tokens = correct_token_predictions & output_mask
        total_correct_tokens = mask_correct_tokens.sum()
        total_tokens = output_mask.sum()

        # Sample-level accuracy
        # Check if all tokens in a sequence are correct for each sample
        correct_program_mask = output_mask.sum(axis=1) == mask_correct_tokens.sum(axis=1)
        total_samples = y.shape[0]

        token_accuracy = total_correct_tokens / total_tokens
        sample_accuracy = correct_program_mask.sum() / total_samples

        return loss, sample_accuracy, token_accuracy
    


    def learn(self, train_examples, iters=8, combined=False, num_epochs=1000, desired_norm=1.0, lr=0.01):
        assert isinstance(train_examples, list)
        assert all(isinstance(example, tuple) for example in train_examples)
        assert all(len(example) == 2 for example in train_examples)

        trainabled_params = [param for param in self._model.parameters() if param.requires_grad]
        optimizer = LazyAdamW(trainabled_params, lr=lr, weight_decay=0.00, betas=(0.9, 0.95), eps=1e-8)
        train_batches = self.example_to_batch(train_examples, combined=combined)

        combined_batch = self.example_to_batch(train_examples, combined=True)[0]
        for epoch in range(num_epochs):
            random.shuffle(train_batches)
            for batch in train_batches:
                (p, i, l), (y, y_l) = batch
                optimizer.zero_grad()
                logits, _, _  = self._model(p, i, l, iters)
                loss = self._model.loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                # Normalize the embedding
                if desired_norm is not None:
                    current_norm = torch.norm(self._model.pte.weight, p=2)
                    desired_norm = 1.0
                    scaling_factor = desired_norm / current_norm
                    self._model.pte.weight.data *= scaling_factor

            
            loss, sample_accuracy, token_accuracy = self._evaluate(combined_batch)
            print(f'Epoch: {epoch}, Loss: {loss:4f}, Sample Accuracy: {sample_accuracy:4f}, Token Accuracy: {token_accuracy:4f}')
            if sample_accuracy == 1.0:
                break

    @torch.no_grad()
    def solve(self, test_examples):
        for example in test_examples:
            (p, i), o = self.tokenize(example)

            predictions = self._model.beam_search(p, i, 8, max_length=1024, top_k=10, eos_token_id=12)
            # print(p, i, o)
            print("Output:", o)
            for pred, score in predictions:
                print("Pred:", pred)
                if pred == o:
                    print("Correct")
                    break


#%%
from src.task import TRAINING_TASKLOADER, EVALUATION_TASKLOADER

# checkpoint_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/lightning_runs/runs/A2D5M512H16B5L8.v4s/checkpoints/checkpoint_219670.pth'
# checkpoint_path = '/teamspace/studios/this_studio/ArcSolveR/runs/V5_11Sept/A2D5M256H8B4L8_v15/checkpoints/checkpoint_420211.pth'
checkpoint_path = '/teamspace/studios/this_studio/ArcSolveR/runs/A2D5M512H16B5L8.v2.3/checkpoint_177891.pth'
solver = ArcSolver(checkpoint_path)
solver.reset()
tasks = TRAINING_TASKLOADER.load_tasks(None)
# tasks = EVALUATION_TASKLOADER.load_tasks(None)
#%%
task = tasks[22]
print("Loaded Task:", task.id)
solver.reset()
solver.learn(task.train, iters=8, num_epochs=1000, desired_norm=1.0, combined=False, lr=0.01)

#%%
solver.learn(task.train, iters=8, num_epochs=1000, desired_norm=1.0, combined=False, lr=0.005)

# %%
solver.solve(task.test)
# %%
import random