import math
import time
import torch
from .dataset import GridTokenizer, ProgramTokenizer
from .interpreter import Interpreter, InterpreterConfig
from .trainer import TrainerBase
# from .trainer_copy import TrainerBase

class ArcTrainer(TrainerBase):

    @staticmethod
    def _output_target_metrics(logits: torch.Tensor, y: torch.Tensor):
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        total_correct_tokens = correct_token_predictions.sum()
        total_tokens = y.numel()

        # Sample-level accuracy
        # Check if all tokens in a sequence are correct for each sample
        correct_samples = correct_token_predictions.all(dim=1)
        total_correct_samples = correct_samples.sum()
        total_samples = y.shape[0]

        return {
            'total_correct_tokens': total_correct_tokens.item(),
            'total_tokens': total_tokens,
            'total_correct_samples': total_correct_samples.item(),
            'total_samples': total_samples
        }
    
    def _add_step_metrics(self, metrics_obj, loss, logits, y):
        batch_metrics = self._output_target_metrics(logits, y)
        metrics_obj.add_metric('Loss', loss.item())
        metrics_obj.add_metric('TokenAcc(%)',
                            batch_metrics['total_correct_tokens']*100, 
                            batch_metrics['total_tokens'])
        
        metrics_obj.add_metric('BatchSize(#Tokens)', batch_metrics['total_tokens'])
        metrics_obj.add_metric('SampleAcc(%)',
                            batch_metrics['total_correct_samples']*100,
                            batch_metrics['total_samples'])

    def pre_train_step(self, batch):
        self.__train_batch_time_start = time.time()

    def pre_eval_step(self, batch):
        self.__eval_batch_time_start = time.time()
    
    def train_step(self, batch):
        (p, i), t = batch
        logits = self.model(p, i)
        loss = self.model.loss_fn(logits, t)
        self._add_step_metrics(self.train_metrics, loss, logits, t)
        return loss
    
    def eval_step(self, batch):
        (p, i), t = batch
        logits = self.model(p, i)
        loss = self.model.loss_fn(logits, t)    
        self._add_step_metrics(self.eval_metrics, loss, logits, t)
        return loss
    
    def post_train_step(self, batch):
        (_, _), t = batch
        num_tokens = t.numel()
        train_batch_time = (time.time() - self.__train_batch_time_start)*1000
        self.train_metrics.add_metric('ΔT(ms)', train_batch_time)
        self.train_metrics.add_metric('#TokensPerSec', num_tokens, (train_batch_time / 1000))

        if torch.cuda.is_available():
            torch.cuda.synchronize() # wait for the GPU to finish work
        elif self.device.type == 'mps':
            torch.mps.synchronize() # wait for the MPS to finish work
            torch.mps.empty_cache() # clear the MPS cache

    def post_eval_step(self, batch):
        (_, _), t = batch
        num_tokens = t.numel()
        eval_batch_time = (time.time() - self.__eval_batch_time_start)*1000
        self.eval_metrics.add_metric('ΔT(ms)', eval_batch_time)
        self.eval_metrics.add_metric('#TokensPerSec', num_tokens, (eval_batch_time / 1000))

    def state_dict(self):
        state_dict = super().state_dict()
        tokenizers = {
            'program_tokenizer': self.model.prog_tokenizer.to_dict(),
            'grid_tokenizer': self.model.grid_tokenizer.to_dict()
        }
        state_dict['tokenizers'] = tokenizers
        state_dict['model_config'] = self.model.config.to_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
        grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
        model_config = InterpreterConfig.from_dict(state_dict['model_config'])
        assert model_config == self.model.config, "Cannot resume, Model Configs do not match!"
        assert prog_tokenizer == self.model.prog_tokenizer, "Cannot resume, Program Tokenizers do not match!"
        assert grid_tokenizer == self.model.grid_tokenizer, "Cannot resume, Grid Tokenizers do not match!"
        return super().load_state_dict(state_dict)
       

    @staticmethod
    def load_model_from_checkpoint(checkpoint_path, device='cpu'):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_config = InterpreterConfig.from_dict(state_dict['model_config'])
        prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
        grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
        model = Interpreter(model_config, prog_tokenizer, grid_tokenizer)
        model.load_state_dict(state_dict['model_state_dict'])
        return model.to(device)