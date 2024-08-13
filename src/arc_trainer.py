from collections import defaultdict, Counter
import logging
import math
from pathlib import Path
import time
from typing import Dict, Tuple, Union
import numpy as np
import pandas as pd
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

        # import ipdb; ipdb.set_trace()        
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



class SampleStats:
    def __init__(self) -> None:
        self.sample_stats: Dict[str, Stats] = {}

    def add(self, stat_name, key, correct=False):
        if stat_name not in self.sample_stats:
            self.sample_stats[stat_name] = Stats()
        self.sample_stats[stat_name].add(key, correct)

    def table(self, stat_name):
        return  self.sample_stats[stat_name].table()
    
    def histogram(self, stat_name, correct=False, num_bins=20):
        return self.sample_stats[stat_name].histogram(correct, num_bins)


class Stats:
    def __init__(self) -> None:
        self.data_corrects = defaultdict(int)
        self.data_total = defaultdict(int)
        self.non_zero_total_keys = set()
        self.non_zero_correct_keys = set()

    def add(self, key, correct=False):
        self.data_total[key] += 1
        self.data_corrects[key] += (1 if correct else 0)
        self.non_zero_total_keys.add(key)
        if correct:
            self.non_zero_keys.add(key)

    def table(self):
        data = []
        for key in self.data_total:
            data.append((key, self.data_corrects[key] / self.data_total[key], self.data_corrects[key], self.data_total[key]))
        table = wandb.Table(data=data, columns=["key", "accuracy", "num_correct", "num_total"])
        return table
    
    def histogram(self, correct=False, num_bins=20):
        data_dict = self.data_corrects if correct else self.data_total
        key_set = self.non_zero_correct_keys if correct else self.non_zero_total_keys
        # Define your bin edges manually or use np.linspace, np.arange, etc.

        data_min = min(data_dict.keys())
        data_max = max(data_dict.keys())
        bin_edges = np.linspace(data_min, data_max, num_bins + 1)

        # Initialize histogram counts to zero
        hist = np.zeros(num_bins)

        # Iterate over the dictionary and add frequencies to the appropriate bins
        for value in key_set:
            frequency = data_dict[value]

            # Find the index of the bin to which the value belongs
            bin_index = np.digitize(value, bin_edges, right=True) - 1
            # Make sure the value is within the range of our bins
            bin_index = min(bin_index, num_bins - 1)
            # Add frequency to the corresponding bin
            hist[bin_index] += frequency

        return hist, bin_edges
        


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


class ArcTrainer(TrainerBase):

    def at_training_start(self):
        wandb.watch(self.model, log='all', log_freq=len(self.train_dl)//4)
        self.embd_token_indices, self.embedding_datasets = init_embedding_proj(self.model.prog_tokenizer.token2idx)
        def log_dataset_stats(dl, suffix="train"):
            ranks = []
            datasets = Counter()
            versions = Counter()
            dataset_versions = Counter()
            for i, batch in enumerate(dl):
                _, _, meta = batch
                ranks += [m['rank'] for m in meta]
                datasets.update([m['dataset'] for m in meta])
                versions.update([m['version'] for m in meta])
                dataset_versions.update([f"{meta[i]['dataset']}_{meta[i]['version']}" for i in range(len(meta))])


            dataset_table = wandb.Table(data=sorted([(k, v) for k, v in datasets.items()]) ,columns=["key", "num_total"])
            version_table = wandb.Table(data=sorted([(k, v) for k, v in versions.items()]) ,columns=["key", "num_total"])
            dataset_version_table = wandb.Table(data=sorted([(k, v) for k, v in dataset_versions.items()]) ,columns=["key", "num_total"])

            ranks = [[r] for r in ranks]
            rank_table = wandb.Table(data=ranks, columns=["rank"])

            wandb.log({
                f'dataset_samples/{suffix}': wandb.plot.bar(dataset_table, "key", "num_total", title=f"Sample Distribution per Dataset/{suffix}"),
                f'version_samples/{suffix}': wandb.plot.bar(version_table, "key", "num_total", title=f"Sample Distribution per Version/{suffix}"),
                f'dataset_version_samples/{suffix}': wandb.plot.bar(dataset_version_table, "key", "num_total", title=f"Sample Distribution per Dataset-Version/{suffix}"),
                f'rank_samples/{suffix}': wandb.plot.histogram(rank_table, "rank", title=f"Rank Distribution/{suffix}"),
                },
                commit=False
                )

        log_dataset_stats(self.train_dl, "train")
        log_dataset_stats(self.eval_dl, "eval")


    @staticmethod
    def _epoch_end_log(stats, step, suffix="train"):
        dataset_table = stats.table('dataset')
        version_table = stats.table('version')
        dataset_version_table = stats.table('dataset_version')
        rank_correct_hist = stats.histogram('rank', correct=True)
        wandb.log({
                    f'dataset_accuracy/{suffix}': wandb.plot.bar(dataset_table, "key", "accuracy", title=f"Sample Accuracy per Datase/{suffix}"),
                    f'dataset_num_correct/{suffix}': wandb.plot.bar(dataset_table, "key", "num_correct", title=f"Number Correct Sample Predictions/{suffix}"),
                    f'version_accuracy/{suffix}': wandb.plot.bar(version_table, "key", "accuracy", title=f"Sample Accuracy per Version/{suffix}"),
                    f'version_num_correct/{suffix}': wandb.plot.bar(version_table, "key", "num_correct", title=f"Number Correct Sample Predictions/{suffix}"),
                    f'dataset_version_accuracy/{suffix}': wandb.plot.bar(dataset_version_table, "key", "accuracy", title=f"Sample Accuracy per Dataset_Version/{suffix}"),
                    f'dataset_version_num_correct/{suffix}': wandb.plot.bar(dataset_version_table, "key", "num_correct", title=f"Number Correct Sample Predictions/{suffix}"),
                    f'rank_corrects/{suffix}': wandb.Histogram(np_histogram=rank_correct_hist)
                },
                step=step)


    def at_epoch_start(self):
        self.train_stats = SampleStats()

    def at_epoch_end(self):
        if self.disable_checkpointing_and_logging:
            return

        self._epoch_end_log(self.train_stats, self.step, suffix="train")

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
        self.eval_stats = SampleStats()

    def at_eval_end(self):
        if self.disable_checkpointing_and_logging:
            return
        self._epoch_end_log(self.eval_stats, self.step, suffix="eval")


    @staticmethod
    def _output_target_metrics(logits: torch.Tensor, y: torch.Tensor, stats, meta: dict):
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        total_correct_tokens = correct_token_predictions.sum()
        total_tokens = y.numel()

        # Sample-level accuracy
        # Check if all tokens in a sequence are correct for each sample
        correct_samples = correct_token_predictions.all(dim=1)

        for idx in range(len(correct_samples)):
            meta_idx = meta[idx]
            is_test = meta_idx['is_test']
            is_correct = correct_samples[idx] != 0
            for key in ['rank', 'dataset', 'version']:
                stats.add(key, meta_idx[key], is_correct)        
            stats.add('dataset_version', f"{meta_idx['dataset']}_{meta_idx['version']}", is_correct)
        # correct_sample_indices = correct_samples.nonzero(as_tuple=True)[0]

        total_correct_samples = correct_samples.sum()
        total_samples = y.shape[0]

        return {
            'total_correct_tokens': total_correct_tokens.item(),
            'total_tokens': total_tokens,
            'total_correct_samples': total_correct_samples.item(),
            'total_samples': total_samples
        }
    
    def _add_step_metrics(self, metrics_obj, loss, logits, y, stats_obj, meta):
        batch_metrics = self._output_target_metrics(logits, y, stats_obj, meta)
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
        (p, i), t, m = batch
        logits = self.model(p, i)
        loss = self.model.loss_fn(logits, t)
        if not self.disable_checkpointing_and_logging:
            self._add_step_metrics(self.train_metrics, loss, logits, t, self.train_stats, m)
        return loss
    
    def eval_step(self, batch):
        (p, i), t, m = batch
        logits = self.model(p, i)
        loss = self.model.loss_fn(logits, t)    
        if not self.disable_checkpointing_and_logging:
            self._add_step_metrics(self.eval_metrics, loss, logits, t, self.eval_stats, m)
        return loss
    
    def post_train_step(self, batch):
        (_, _), t, m = batch
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
        (_, _), t, m = batch
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