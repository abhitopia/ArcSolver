#%%
import json
import numpy as np
import torch
from tqdm import tqdm
from src.arc_trainer import ArcTrainer
from src.dataset import GridTokenizer, ProgramTokenizer, TrainingData, TaskToExamples
from src.interpreter import Interpreter, InterpreterConfig
from src.task import AUXILIARY_TASKLOADERS, TRAINING_TASKLOADER

# %%

# %%
tasks = TRAINING_TASKLOADER.load_tasks(augmentation_id=0)
# %%

def split_1fold(samples):
    # This will hold our pairs of training and validation sets
    folds = []
    # Create the pairs
    for i in range(len(samples)):
        validation_set = [samples[i]]  # The validation set is the current sample
        training_set = samples[:i] + samples[i+1:]  # The training set is all other samples
        folds.append((training_set, validation_set))

    return folds


class InferenceEngine:
    def __init__(self, checkpoint_path: str):
        self.model = self.load_checkpoint(checkpoint_path)
        self.prog_tokenizer = self.model.prog_tokenizer
        self.grid_tokenizer = self.model.grid_tokenizer
        self.prog_embedding = torch.zeros((1, self.model.config.n_prog_embd), requires_grad=True)
        


    @staticmethod
    def load_checkpoint(checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
        grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
        model_config = InterpreterConfig.from_dict(state_dict['model_config'])

        checkpoint_model = Interpreter(model_config, prog_tokenizer, grid_tokenizer)
        checkpoint_model.load_state_dict(state_dict['model_state_dict'])
        return checkpoint_model
    

    def create_batch(self, examples, seq_len=1024):
        inputs = []
        targets = []
        programs = []
        for example in examples:
            (p, i), o = example
            inp = self.grid_tokenizer.encode(i)
            out = self.grid_tokenizer.encode(o)

            inp = inp + [self.grid_tokenizer.PAD_IDX] * (seq_len - len(inp))
            out = out + [self.grid_tokenizer.PAD_IDX] * (seq_len - len(out))            
            prog = self.prog_tokenizer.encode(p) if self.prog_tokenizer else None

            inputs.append(inp)
            targets.append(out)
            programs.append(prog)

        inputs = torch.tensor(inputs, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        programs = programs[0] if self.prog_tokenizer else None
        return (programs, inputs), targets
    

    def metrics(self, logits, targets):
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == targets)
        total_correct_tokens = correct_token_predictions.sum()
        total_tokens = targets.numel()

        correct_samples = correct_token_predictions.all(dim=1)
        total_correct_samples = correct_samples.sum()
        total_samples = targets.shape[0]

        token_accuracy = total_correct_tokens / total_tokens
        sample_accuracy = total_correct_samples / total_samples
        return token_accuracy.item(), sample_accuracy.item()

    def evaluate(self, batch):
        self.model.eval()
        (p, i), o = batch
        batch_size = i.shape[0]
        prog_tensor = self.prog_embedding.expand(batch_size, -1, -1)    
        with torch.no_grad():
            logits = self.model.infer(prog_tensor, i)
            loss = self.model.loss_fn(logits, o)
        token_accuracy, sample_accuracy = self.metrics(logits, o)
        return loss.item(), token_accuracy, sample_accuracy
    

    def train(self, batch, lr=0.1, train_model=False):
        self.model.eval()
        (p, i), o = batch
        batch_size = i.shape[0]
        prog_tensor = self.prog_embedding.expand(batch_size, -1, -1)
        logits = self.model.infer(prog_tensor, i)
        loss = self.model.loss_fn(logits, o)

        if train_model:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
            gradients = torch.autograd.grad(loss, self.model.parameters(), create_graph=True, allow_unused=True)
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), gradients):
                    if grad is None:
                        continue
                    param -= lr * grad
        else:
            gradients = torch.autograd.grad(loss, self.prog_embedding)[0]
            with torch.no_grad():  # Update weights without tracking gradients
                self.prog_embedding -= lr * gradients

        token_accuracy, sample_accuracy = self.metrics(logits, o)
        return loss.item(), token_accuracy, sample_accuracy

    def infer(self, task, num_iter=10, init_emb=True, lr=0.1, train_model=False):
        train_examples, test_examples = TaskToExamples()(task)
        folds = split_1fold(train_examples)
        if init_emb:
            program = [task.id, task.version]
            prog_str = '_'.join(program)
            prog_idx = self.prog_tokenizer.encode(prog_str)[0]
            init_emb = self.model.pte(torch.tensor([prog_idx], dtype=torch.long).reshape(1,))
            self.prog_embedding.data[:] = init_emb.data[:]

        for it in range(num_iter):
            for fid, (t, e) in enumerate(folds):
                t_batch = self.create_batch(t)
                e_batch = self.create_batch(e)

                t_loss, t_t_acc, t_s_acc = self.train(t_batch, lr=lr, train_model=train_model)
                e_loss, e_t_acc, e_s_acc = self.evaluate(e_batch)

                print(f'Iter: {it}, Fold: {fid}')
                print(f'Train Loss: {t_loss:.4f}, Train Token Acc: {t_t_acc:.4f}, Train Sample Acc: {t_s_acc:.4f}')
                print(f'Eval Loss: {e_loss:.4f}, Eval Token Acc: {e_t_acc:.4f}, Eval Sample Acc: {e_s_acc:.4f}')
          
# %%
# %%
checkpoint_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/runs/D10M16.5.1.5/v32/checkpoints/checkpoint_037779.pth'
engine = InferenceEngine(checkpoint_path)
engine.infer(tasks[0], num_iter=10, init_emb=True, lr=0.5)
engine.infer(tasks[0], num_iter=10, init_emb=False, lr=0.1)
engine.infer(tasks[0], num_iter=10, init_emb=False, lr=0.05)
# %%

# %%
engine.infer(tasks[0], num_iter=10, init_emb=False, lr=0.1)

# %%
# %%

engine.infer(tasks[0], num_iter=50, init_emb=False, lr=0.01, train_model=True)
# %%
engine.infer(tasks[0], num_iter=30, init_emb=False, lr=0.001, train_model=False)
# %%
