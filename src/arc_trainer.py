import math
import time
import torch
from .dataset import GridTokenizer, ProgramTokenizer
from .interpreter import Interpreter, InterpreterConfig
from .trainer import TrainerBase

class ArcTrainer(TrainerBase):

    @staticmethod
    def _output_target_metrics(logits: torch.Tensor, y: torch.Tensor):
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        total_correct_tokens = correct_token_predictions.sum()
        total_tokens = y.numel()
        # token_accuracy = total_correct_tokens / total_tokens

        # Sample-level accuracy
        # Check if all tokens in a sequence are correct for each sample
        correct_samples = correct_token_predictions.all(dim=1)
        total_correct_samples = correct_samples.sum()
        total_samples = y.shape[0]
        # sample_accuracy = total_correct_samples / total_samples

        return {
            'total_correct_tokens': total_correct_tokens.item(),
            'total_tokens': total_tokens,
            'total_correct_samples': total_correct_samples.item(),
            'total_samples': total_samples
        }
    
    def add_step_metrics(self, metrics_obj, loss, logits, y):
        batch_metrics = self._output_target_metrics(logits, y)
        metrics_obj.add_metric('Loss', loss.item())
        metrics_obj.add_metric('TokenAcc(%)',
                            batch_metrics['total_correct_tokens']*100, 
                            batch_metrics['total_tokens'])
        
        metrics_obj.add_metric('BatchSize(#Tokens)', batch_metrics['total_tokens'])
        metrics_obj.add_metric('SampleAcc(%)',
                            batch_metrics['total_correct_samples']*100,
                            batch_metrics['total_samples'])

    def add_post_step_metrics(self, metrics_obj, batch_time, num_tokens):
        # Batch Metrics
        metrics_obj.add_metric('#TokensPerSec', num_tokens, (batch_time / 1000))
        metrics_obj.add_metric('ΔT(ms)', batch_time)

    def pre_train_step(self, batch):
        self.__train_batch_time_start = time.time()

    def pre_eval_step(self, batch):
        self.__eval_batch_time_start = time.time()
    
    def train_step(self, batch):
        (p, i), t = batch
        logits = self.model(p, i)
        loss = self.model.loss_fn(logits, t)
        self.add_step_metrics(self.train_metrics, loss, logits, t)
        return loss
    
    def eval_step(self, batch):
        (p, i), t = batch
        logits = self.model(p, i)
        loss = self.model.loss_fn(logits, t)    
        self.add_step_metrics(self.eval_metrics, loss, logits, t)
        return loss
    
    def post_train_step(self, batch):
        (_, _), t = batch
        num_tokens = t.numel()
        train_batch_time = (time.time() - self.__train_batch_time_start)*1000
        self.add_post_step_metrics(self.train_metrics, train_batch_time, num_tokens)

        if torch.cuda.is_available():
            torch.cuda.synchronize() # wait for the GPU to finish work
        elif self.device.type == 'mps':
            torch.mps.synchronize() # wait for the MPS to finish work
            torch.mps.empty_cache() # clear the MPS cache

    def post_eval_step(self, batch):
        (_, _), t = batch
        num_tokens = t.numel()
        eval_batch_time = (time.time() - self.__eval_batch_time_start)*1000
        self.add_post_step_metrics(self.eval_metrics, eval_batch_time, num_tokens)

    def pre_checkpoint_save(self, state_dict):
        tokenizers = {
            'program_tokenizer': self.model.prog_tokenizer.to_dict(),
            'grid_tokenizer': self.model.grid_tokenizer.to_dict()
        }
        state_dict['tokenizers'] = tokenizers
        state_dict['model_config'] = self.model.config.to_dict()

    def post_load_checkpoint(self, state_dict):
        prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
        grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
        model_config = InterpreterConfig.from_dict(state_dict['model_config'])
        assert model_config == self.model.config, "Cannot resume, Model Configs do not match!"
        assert prog_tokenizer == self.model.prog_tokenizer, "Cannot resume, Program Tokenizers do not match!"
        assert grid_tokenizer == self.model.grid_tokenizer, "Cannot resume, Grid Tokenizers do not match!"

    def multiplicative_lr_factor_schedule(self, step):
        max_lr = 1.0
        min_lr = max_lr * 0.05
        num_step_in_epoch = len(self.train_dl)
        warmup_steps = num_step_in_epoch * self.hparams['lr_warmup_epochs']
        max_steps = num_step_in_epoch * self.hparams['lr_decay_epochs']

        # 1) linear warmup for warmup_iters steps
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if step > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
        

    @staticmethod
    def load_model_from_checkpoint(checkpoint_path, device='cpu'):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_config = InterpreterConfig.from_dict(state_dict['model_config'])
        prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
        grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
        model = Interpreter(model_config, prog_tokenizer, grid_tokenizer)
        model.load_state_dict(state_dict['model_state_dict'])
        return model.to(device)