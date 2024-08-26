from collections import defaultdict, Counter
import logging
import math
from pathlib import Path
import time
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
import torch
from .dataset import GridTokenizer, ProgramTokenizer, AUXILIARY_TASKLOADERS
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
import wandb



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


def const_schedule(step, warmup_steps):
    max_lr = 1.0
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    return max_lr

def lin_decay_schedule(step, warmup_steps, max_steps, min_lr_scale=0.1):
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
    # 3) in between, use linear decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    return max_lr - decay_ratio * (max_lr - min_lr)

def get_alt_schedulers(num_steps_in_epoch):
    first_scheduler = lambda step: 1.0 if (step // num_steps_in_epoch) % 2 == 0 else 0.0
    second_scheduler = lambda step: 0.0 if (step // num_steps_in_epoch) % 2 == 0 else 1.0
    return first_scheduler, second_scheduler


class ArcHparams(Hparams):

    def init_dataloaders(self)-> Tuple[DataLoader, DataLoader]:

        training_data = TrainingData(
                    augmentation_factor=self.data.data_aug,
                    join_version=True,
                    auxilliary_loader=AUXILIARY_TASKLOADERS if self.data.use_aux else [],
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
                                           seq_len=config.batch_seq_len,
                                           batch_by_token_count=True,
                                           pin_memory=True,
                                           shuffle=True)

        eval_dl = eval_ds.get_dataloader(batch_size=config.batch_size,
                                        seq_len=config.batch_seq_len,
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

        optimizer = model.get_optimizer(
                                    model_lr=config.lr_model,
                                    model_wd=config.wd_model,
                                    prog_lr=config.lr_prog,
                                    prog_wd=config.wd_prog,
                                    device_type=self.device)

        return optimizer
    
    def init_scheduler(self, optimizer)-> optim.lr_scheduler.LambdaLR:
        config = self.optim

        warmup_steps = config.lr_warmup_steps
        max_steps = config.lr_warmup_steps +  config.lr_decay_steps
        if config.lr_schedule == 'noam': 
            schedule = lambda step: noam_schedule(step, warmup_steps, max_steps)
        elif config.lr_schedule == 'lindecay':
            schedule = lambda step: lin_decay_schedule(step, warmup_steps, max_steps)
        elif config.lr_schedule == 'const':
            schedule = lambda step: const_schedule(step, warmup_steps)
        elif config.lr_schedule == 'alt':
            assert len(optimizer.param_groups) == 3, "Invalid LR Schedule"
            high_low_schedule, low_high_schedule = get_alt_schedulers(config.lr_decay_steps)
            schedule = [high_low_schedule, low_high_schedule, low_high_schedule]
        else:
            raise ValueError(f"Invalid LR Schedule: {config.lr_schedule}. Options: noam, const, alt")

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
        scheduler._step_count = -1 # To prevent warning because initialation makes a first call to step automatically
        return scheduler


def init_embedding_proj(token2idx):
    synth_datasets = {'_'.join(token.split('_')[1:3]) for token in token2idx if 'SYNTH' in token}
    sample_counter = Counter({dataset: 0 for dataset in synth_datasets})

    total_points = 1000
    points_per_dataset = total_points // len(synth_datasets)

    embedding_tokens = set()

    for token in token2idx:
        if 'SYNTH' in token:
            dataset = '_'.join(token.split('_')[1:3])
            if sample_counter[dataset]  < points_per_dataset:
                sample_counter[dataset] += 1
                embedding_tokens.add(token)

    embedding_tokens = list(embedding_tokens)
    embedding_datasets = [token.split("_")[2] for token in embedding_tokens]
    embd_token_indices = [token2idx[token] for token in embedding_tokens]

    return embd_token_indices, embedding_datasets


class DatasetLevelStats:
    def __init__(self, prog_level_map, prog_tokenizer, suffix):
        self.prog_level_map = prog_level_map
        self.prog_tokenizer = prog_tokenizer
        self.suffix = suffix
        self.prog2level = np.zeros(len(self.prog_tokenizer), dtype=int)
        self.prog2dataset = np.zeros(len(self.prog_tokenizer), dtype=int)
        for level, progs in self.prog_level_map.items():
            for prog in progs:
                self.prog2level[prog] = level-1


        idx2token = self.prog_tokenizer.idx2token
        dataset2progs = defaultdict(set)

        for pid in range(len(idx2token)):
            token = idx2token[pid]
            split_token = token.split('_')
            third = split_token[2]
            last_dataset_token_idx = 3 if third.isupper() else 2
            dataset = '_'.join(split_token[:last_dataset_token_idx])
            dataset2progs[dataset].add(pid)

        self.datasets = list(dataset2progs.keys())

        for dataset, progs in dataset2progs.items():
            for prog in progs:
                self.prog2dataset[prog] = self.datasets.index(dataset)

        self.num_levels = len(self.prog_level_map.keys())
        self.num_progs = len(self.prog_tokenizer)
        self.num_datasets = len(self.datasets)

        self.totals = np.zeros((self.num_levels, self.num_datasets), dtype=int)
        self.corrects = np.zeros((self.num_levels, self.num_datasets), dtype=int)

    def epoch_reset(self):
        self.corrects = np.zeros((self.num_levels, self.num_datasets))

    def update_totals(self, prog_indices):
        assert prog_indices.ndim == 1
        all_prog_levels = self.prog2level[prog_indices]
        all_prog_datasets = self.prog2dataset[prog_indices]
        np.add.at(self.totals, (all_prog_levels, all_prog_datasets), 1)

    def update_corrects(self, prog_indices):
        correct_prog_levels = self.prog2level[prog_indices]
        correct_prog_datasets = self.prog2dataset[prog_indices]
        np.add.at(self.corrects, (correct_prog_levels, correct_prog_datasets), 1)
        
    def get_accuracy(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            accuracy = np.divide(self.corrects, self.totals, where=self.totals!=0)
            accuracy[self.totals == 0] = 0
            return accuracy

    def get_accuracy_by_level(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            totals = self.totals.sum(axis=1)
            accuracy = np.divide(self.corrects.sum(axis=1), totals, where=totals!=0)
            accuracy[totals == 0] = 0
            return accuracy


    def get_accuracy_by_dataset(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            totals = self.totals.sum(axis=0)
            accuracy = np.divide(self.corrects.sum(axis=0), totals, where=totals!=0)
            accuracy[totals == 0] = 0
            return accuracy

    def log_totals(self):
        for level in range(self.num_levels):
            data = [(self.datasets[did], self.totals[level, did]) for did in range(self.num_datasets)]
            dataset_table = wandb.Table(data=data ,columns=["dataset", "total"])
            wandb.log({f'totals/level_{level+1}/{self.suffix}': wandb.plot.bar(dataset_table, "dataset", "total", title=f"({self.suffix}) Totals at Level {level+1}")}, commit=False)


    def log_accuracy(self, step):
        accuracy = self.get_accuracy()
        datasets_accuracy = self.get_accuracy_by_dataset()
        level_accuracy = self.get_accuracy_by_level()

        metrics = {}

        for level in range(self.num_levels):
            metrics[f'LevelAccuracy/{self.suffix}/level_{level+1}'] = level_accuracy[level]

            for did in range(self.num_datasets):
                if level == 0:
                    metrics[f'DatasetAccuracy/{self.suffix}/{self.datasets[did]}'] = datasets_accuracy[did]

                metrics[f'AccuracyDL/{self.suffix}/{self.datasets[did]}_L_{level+1}'] = accuracy[level, did]

        # wandb.plot.line_series(metrics, step=step)
        wandb.log(metrics, step=step)


class ArcTrainer(TrainerBase):

    def at_training_start(self):
        self.checkpoint_metric = 'SampleAcc(%)'
        self.checkpoint_metric_increases = True
        
        # Log params every 10 epochs
        wandb.watch(self.model, log='all', log_freq=max(len(self.train_dl)*10, 500)) 

        if self.hparams.data.use_aux:
            self.embd_token_indices, self.embedding_datasets = init_embedding_proj(self.model.prog_tokenizer.token2idx)

        self.train_stats = DatasetLevelStats(self.train_dl.prog_level_map, self.model.prog_tokenizer, "train")
        self.eval_stats = DatasetLevelStats(self.eval_dl.prog_level_map, self.model.prog_tokenizer, "eval")

        def log_dataset_stats(dl, stats):
            for i, batch in enumerate(dl):
                (p, i), t = batch
                program_indices = p[:, 0].detach().cpu().numpy()
                stats.update_totals(program_indices)

            stats.log_totals()

        log_dataset_stats(self.train_dl, self.train_stats)
        log_dataset_stats(self.eval_dl, self.eval_stats)


    def at_epoch_start(self):
        self.train_stats.epoch_reset()

    def at_epoch_end(self):
        if self.disable_checkpointing_and_logging:
            return

        self.train_stats.log_accuracy(self.step)

        if self.epoch % 50 == 0 and self.hparams.data.use_aux:
            # Only log the embeddings every 10 epochs
            embd_weight = self.model.pte.weight
            indices = torch.tensor(self.embd_token_indices,
                                dtype=torch.long,
                                device=embd_weight.device)
            subset_embeddings = embd_weight[indices]
            subset_embeddings_np = subset_embeddings.cpu().detach().numpy()

            df = pd.DataFrame(
                        subset_embeddings_np,
                        index=self.embedding_datasets)
            df.reset_index(inplace=True)
            df.columns = ['dataset'] + [f'dim_{i}' for i in range(subset_embeddings_np.shape[1])]

            wandb.log({'dataset_embedding': df},
                commit=False)


    def at_eval_start(self):
        self.eval_stats.epoch_reset()
    
    def at_eval_end(self):
        if self.disable_checkpointing_and_logging:
            return

        self.eval_stats.log_accuracy(self.step)


    def _output_target_metrics(self, program_indices: torch.Tensor, logits: torch.Tensor, y: torch.Tensor, is_train):
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        total_correct_tokens = correct_token_predictions.sum()
        total_tokens = y.numel()

        # Sample-level accuracy
        # Check if all tokens in a sequence are correct for each sample
        correct_samples = correct_token_predictions.all(dim=1)

        correct_program_mask = correct_samples.to(dtype=torch.bool)
        correct_program_indices = program_indices[correct_program_mask, 0].cpu().numpy()

        stats = self.train_stats if is_train else self.eval_stats
        stats.update_corrects(correct_program_indices)
        total_correct_samples = correct_samples.sum()
        total_samples = y.shape[0]

        return {
            'total_correct_tokens': total_correct_tokens.item(),
            'total_tokens': total_tokens,
            'total_correct_samples': total_correct_samples.item(),
            'total_samples': total_samples
        }
    
    def _add_step_metrics(self, loss, program_indices, logits, y, is_train):
        metrics_obj = self.train_metrics if is_train else self.eval_metrics
        batch_metrics = self._output_target_metrics(program_indices, logits, y, is_train)
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
        if not self.disable_checkpointing_and_logging:
            self._add_step_metrics(loss, p, logits, t, is_train=True)
        return loss
    
    def eval_step(self, batch):
        (p, i), t = batch
        logits = self.model(p, i)
        loss = self.model.loss_fn(logits, t)    
        if not self.disable_checkpointing_and_logging:
            self._add_step_metrics(loss, p, logits, t, is_train=False)
        return loss
    
    def post_train_step(self, batch):
        (_, _), t = batch
        num_tokens = t.numel()
        train_batch_time = (time.time() - self.__train_batch_time_start)*1000
        self.train_metrics.add_metric('ΔT(ms)', train_batch_time)
        self.train_metrics.add_metric('#TokensPerSec', num_tokens, (train_batch_time / 1000))

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize() # wait for the GPU to finish work
        # elif self.device.type == 'mps':
        #     torch.mps.synchronize() # wait for the MPS to finish work
        #     torch.mps.empty_cache() # clear the MPS cache

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