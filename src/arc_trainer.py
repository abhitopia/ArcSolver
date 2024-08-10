import logging
import math
from pathlib import Path
import time
from typing import Tuple, Union
import torch
from .dataset import GridTokenizer, ProgramTokenizer
from .interpreter import Interpreter, InterpreterConfig
from .trainer import TrainerBase, Hparams
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from src.dataset import TrainingData
from src.interpreter import Interpreter, InterpreterConfig
from src.trainer import Hparams
from src.utils import nearest_greater_power_of_2
import math



def noam_schedule(step, warmup_steps, max_steps, min_lr_scale=0.1):
    max_lr = 1.0
    min_lr = max_lr * min_lr_scale
    # num_step_in_epoch = self.state['num_train_batches']
    # warmup_steps = num_step_in_epoch * config.lr_warmup_epochs
    # max_steps = num_step_in_epoch * config.lr_decay_epochs

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

def get_alt_schedulers(num_steps_in_epoch):
    first_scheduler = lambda step: 1.0 if (step // num_steps_in_epoch) % 2 == 0 else 0.0
    second_scheduler = lambda step: 0.0 if (step // num_steps_in_epoch) % 2 == 0 else 1.0
    return first_scheduler, second_scheduler


class ArcHparams(Hparams):

    def init_dataloaders(self)-> Tuple[DataLoader, DataLoader]:

        training_data = TrainingData(
                    augmentation_factor=self.data.data_aug,
                    join_version=True,
                    num_levels=self.data.num_diff_levels,
                    seed=self.seed).load()
        
        # self.state['training_data'] = training_data
        self.state['prog_tokenizer'] = training_data.program_tokenizer
        self.state['grid_tokenizer'] = training_data.grid_tokenizer
        self.state['prog_vocab_size'] = nearest_greater_power_of_2(len(training_data.program_tokenizer))
        self.state['grid_vocab_size'] = nearest_greater_power_of_2(len(training_data.grid_tokenizer))

        # training_data = self.state['training_data']
        config = self.optim
        train_ds = training_data.train_ds(num_levels=self.data.diff_level).subset(config.max_examples)
        eval_ds = training_data.eval_ds(num_levels=self.data.diff_level).subset(config.max_examples)
        
        train_dl = train_ds.get_dataloader(batch_size=config.batch_size,
                                           seq_len=1024,
                                           batch_by_token_count=True,
                                           pin_memory=True,
                                           shuffle=True)

        eval_dl = eval_ds.get_dataloader(batch_size=config.batch_size,
                                        seq_len=1024,
                                        batch_by_token_count=True,
                                        pin_memory=True,
                                        shuffle=False)
        
        self.state['num_train_batches'] = len(train_dl)
        self.state['num_eval_batches'] = len(eval_dl)
        return train_dl, eval_dl
    
    def init_model(self)-> nn.Module:
        config = InterpreterConfig(
            prog_vocab_size = self.state['prog_vocab_size'],
            grid_vocab_size = self.state['grid_vocab_size'],
            n_prog_embd = self.model.n_prog_embd, # size of the program embedding
            n_head = self.model.n_heads, # number of heads within each self-attention block
            n_rec_block = self.model.n_rec_block, 
            n_rec_layer = self.model.n_rec_layer, 
            n_blocks = self.model.n_blocks
        )
        model = Interpreter(config,
                            prog_tokenizer=self.state['prog_tokenizer'],
                            grid_tokenizer=self.state['grid_tokenizer'])
        
        return model
    
    def init_optimizer(self, model: Interpreter)-> optim.Optimizer:
        config = self.optim

        if config.lr_prog is None:
            config.lr_prog = config.lr_model
            if config.lr_schedule == 'noam':
                plr_scale = 1 if self.data.data_aug <= 0 else 8 * self.data.data_aug
                config.lr_prog = config.lr_model * plr_scale

        optimizer = model.get_optimizer(
                                    model_lr=config.lr_model,
                                    model_wd=config.wd_model,
                                    prog_lr=config.lr_prog,
                                    prog_wd=config.wd_prog,
                                    device_type=self.device)

        return optimizer
    
    def init_scheduler(self, optimizer)-> optim.lr_scheduler.LambdaLR:
        config = self.optim

        if config.lr_schedule == 'noam':
            warmup_steps = self.state['num_train_batches'] * config.lr_warmup_epochs
            max_steps = self.state['num_train_batches'] * config.lr_decay_epochs
            schedule = lambda step: noam_schedule(step, warmup_steps, max_steps)
        elif config.lr_schedule == 'const':
            schedule = lambda step: 1.0
        elif config.lr_schedule == 'alt':
            assert len(optimizer.param_groups) == 3, "Invalid LR Schedule"
            high_low_schedule, low_high_schedule = get_alt_schedulers(self.state['num_train_batches'])
            schedule = [high_low_schedule, low_high_schedule, low_high_schedule]
        else:
            raise ValueError(f"Invalid LR Schedule: {config.lr_schedule}. Options: noam, const, alt")

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
        scheduler._step_count = -1 # To prevent warning because initialation makes a first call to step automatically
        return scheduler


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
        metrics_obj.add_metric('#Samples', batch_metrics['total_samples'])
        metrics_obj.add_metric('SeqLen', y.shape[1])
        

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
    
    def load_state_dict(self, state_dict, resume=True, strict=True):
        prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
        grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
        model_config = InterpreterConfig.from_dict(state_dict['model_config'])

        if resume: # Make sure that the model is exactly the same
            assert model_config == self.model.config, "Cannot resume, Model Configs do not match!"
            assert prog_tokenizer == self.model.prog_tokenizer, "Cannot resume, Program Tokenizers do not match!"
            assert grid_tokenizer == self.model.grid_tokenizer, "Cannot resume, Grid Tokenizers do not match!"
            super().load_state_dict(state_dict, resume=resume, strict=strict)    
        else:
            # We don't want default behavior of loading model state dict. We want special 
            # which is able to copy the weights from a compatible model
            checkpoint_model = Interpreter(model_config, prog_tokenizer, grid_tokenizer)
            checkpoint_model.load_state_dict(state_dict['model_state_dict'])
            self.info("Loading model from checkpoint using load_from_model method")
            self.model.load_from_model(checkpoint_model, strict=strict)

            if prog_tokenizer != self.model.prog_tokenizer or grid_tokenizer != self.model.grid_tokenizer:
                self.warning("Loaded model has different tokenizers than the current model. Loading anyway as the models are compatible.")
                self.warning("If this is not intened, stop and re-evaluate the situation.")

            self._eval_at_start = True
       
    
    @classmethod
    def from_checkpoint(
                cls,
                checkpoint_path: Union[str, Path],
                resume=True,
                logger=None,
                disable_checkpointing_and_logging=False,
                parent_dir=None
            ):
        
        return super(ArcTrainer, cls).from_checkpoint(
                    ArcHparams,
                    checkpoint_path=checkpoint_path,
                    resume=resume,
                    logger=logger,
                    disable_checkpointing_and_logging=disable_checkpointing_and_logging,
                    parent_dir=parent_dir)