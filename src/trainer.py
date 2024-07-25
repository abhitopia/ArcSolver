#%%
import json
import math
from box import Box
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import random
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, Union
from collections import defaultdict
from tqdm import tqdm
from .utils import add_logger, map_to_tensors
from dataclasses import dataclass


class MetricLogger:
    def __init__(self):
        self.reset()

    def add_metric(self, name, numerator, denominator=1.0, tags=[]):
        self._data[name].append((numerator, denominator))
        for tag in tags:
            self._tags[name].add(tag)

    def mean_metrics(self, filter_tag=None):
        metrics = {}
        for name in self._data:
            if filter_tag is None or filter_tag in self._tags[name]:
                metrics[name] = self.mean(name)
        return metrics
    
    def last_metrics(self, filter_tag=None):
        metrics = {}
        for name in self._data:
            if filter_tag is None or filter_tag in self._tags[name]:
                metrics[name] = self.last(name)
        return metrics

    def last(self, name):
        n, d = self._data[name][-1]
        return (n / d)
    
    def mean(self, name):
        data = self._data[name]
        return sum([n for n, d in data]) / sum([d for n, d in data])

    def reset(self):
        self._data = defaultdict(list)
        self._tags = defaultdict(set)



@dataclass
class Hparams:
    experiment: str
    run: str
    num_epochs: Optional[int] = None # Number of epochs to train, None means infinite
    clip_grad_norm: Optional[float] = 1.0 # Clip gradient norm, None means no clipping
    seed: int = 1337  # Seed everything for reproducibility
    device: str = None # Device to use for training, None means automatically determine the best device
    eval_interval: Optional[int] = None # Evaluate every n steps, None means evaluate after every epoch

    def __post_init__(self):
        assert isinstance(self.experiment, str), 'experiment must be a string'
        assert isinstance(self.run, str), 'run must be a string'
        assert self.num_epochs is None or self.num_epochs > 0, 'num_epochs must be None or a positive integer'
        assert self.clip_grad_norm is None or self.clip_grad_norm > 0, 'clip_grad_norm must be None or a positive float'
        assert isinstance(self.seed, int), 'seed must be an integer'
        assert self.device is None or isinstance(self.device, str), 'device must be a string or None'
        assert self.eval_interval is None or self.eval_interval > 0, 'eval_interval must be None or a positive integer'
        self._state = {}

    def add_params(self, prefix='', **kwargs):
        if len(prefix) > 0:
            if not hasattr(self, prefix):
                setattr(self, prefix, Box())

        for k, v in kwargs.items():
            assert isinstance(v, (int, float, str)), f'Value of {k} must be int, float or str'
            assert not k.startswith('_'), f'Parameter name cannot start with _'
            if len(prefix) == 0:
                setattr(self, k, v)
            else:
                prefix_dict = getattr(self, prefix)
                prefix_dict[k] = v
    
    def reset_state(self):
        self._state = {}

    @property
    def state(self):
        return self._state

    def init_dataloaders(self)-> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError('init_data_loaders method must be implemented')

    def init_model(self)-> nn.Module:
        raise NotImplementedError('init_model method must be implemented')
    
    def init_optimizer(self, model)-> optim.Optimizer:
        raise NotImplementedError('init_optimizer method must be implemented')
    
    def init_scheduler(self, optimizer)-> optim.lr_scheduler.LambdaLR:
        raise NotImplementedError('init_scheduler method must be implemented')

    def as_dict(self):
        flat_dict = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and not callable(v) and v is None or isinstance(v, (int, float, str)):
                flat_dict[k] = v
            elif isinstance(v, Box):
                for subk, subv in v.items():
                    flat_dict[f'{k}.{subk}'] = subv

        return flat_dict
    
    @classmethod
    def from_dict(cls, hparams_dict):
        args = {k: v for k, v in hparams_dict.items() if '.' not in k} 
        hparams = cls(**args)
        for k, v in hparams_dict.items():
            if '.' in k:
                prefix, key = k.split('.')
                hparams.add_params(prefix, **{key: v})
        return hparams



class TrainerBase:
    def __init__(self,
                hparams: Hparams,
                log_level=logging.INFO,
                parent_dir: Optional[Union[str, Path]] = None,
                disable_checkpointing_and_logging=False,
                prevent_overwrite=True
            ):
        

        assert isinstance(hparams, Hparams), 'hparams must be an instance of Hparams'
        self.hparams = hparams

        self.log_dir = self.get_log_dir(self.hparams.experiment, self.hparams.run, parent_dir=parent_dir)
        self.checkpoint_dir = self.get_checkpoint_dir(self.hparams.experiment, self.hparams.run, parent_dir=parent_dir)

        if prevent_overwrite:
            assert not self.log_dir.exists(), f'Log directory {self.log_dir} already exists.'
            assert not self.checkpoint_dir.exists(), f'Checkpoint directory {self.checkpoint_dir} already exists.'


        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        add_logger(obj=self,
                log_level=log_level,
                name=self.hparams.run,
                file_path=self.log_dir / f'training.log')

        self.info(f"Hparams: {json.dumps(self.hparams.as_dict(), indent=4)}")
        self.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.disable_checkpointing = disable_checkpointing_and_logging
        if self.disable_checkpointing:
            self.warning(f'It is a trial run. No checkpoints or Tensorboard summaries will be saved!')
            self.warning(f'Resuming from checkpoints still works!')

        self.writer = None
        self.step = 0
        self.epoch = 0 
        self.epoch_step = 0
        
        # From Hparams (Direct)
        self.seed = self.hparams.seed
        self.clip_grad_norm = self.hparams.clip_grad_norm
        self.num_epochs = self.hparams.num_epochs
        self.eval_interval = self.hparams.eval_interval


        # From Hparams (Indirect) - Lazy loading
        self._device = None
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._train_dl = None
        self._eval_dl = None
        

        self.train_metrics = MetricLogger()
        self.eval_metrics = MetricLogger()

        self.hparams.init_dataloaders()


    @property
    def device(self):
        if self._device is None:
            self._device = self._init_device(self.hparams.device)
        return self._device

    @property
    def model(self):
        if self._model is None:
            self._model = self.hparams.init_model()
            self._model.to(self.device)
        return self._model
    
    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = self.hparams.init_optimizer(self.model)
        return self._optimizer
    
    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = self.hparams.init_scheduler(self.optimizer)
        return self._scheduler
    
    @property
    def train_dl(self):
        if self._train_dl is None:
            self._train_dl, self._eval_dl = self.hparams.init_dataloaders()
        return self._train_dl
    
    @property
    def eval_dl(self):
        if self._eval_dl is None:
            self._train_dl, self._eval_dl = self.hparams.init_dataloaders()
        return self._eval_dl
    

    @staticmethod
    def get_log_dir(experiment_name, run_name, parent_dir=None):
        if parent_dir is None:
            import __main__
            calling_script = Path(__main__.__file__)
            parent_dir = calling_script.parent / 'runs'
        log_dir =  parent_dir / experiment_name / run_name
        return log_dir

    @staticmethod
    def get_checkpoint_dir(experiment_name, run_name, parent_dir=None):
        log_dir = TrainerBase.get_log_dir(experiment_name, run_name, parent_dir=parent_dir)
        checkpoint_dir = log_dir / 'checkpoints'
        return checkpoint_dir


    @staticmethod
    def _init_device(_device):
        if _device is None:
            return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(_device, str):
            return torch.device(_device)
        elif isinstance(_device, torch.device):
            return _device

    def _seed_everything(self):
        self.logger.info(f'Seeding everything with seed: {self.seed}')
        seed = self.seed
        random.seed(seed)                       # Python's built-in random module
        np.random.seed(seed)                    # NumPy's random module
        torch.manual_seed(seed)                 # PyTorch's random number generator for CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)         # PyTorch's random number generator for CUDA
            torch.cuda.manual_seed_all(seed)     # for multi-GPU setups
            torch.backends.cudnn.deterministic = True  # To increase reproducibility on GPU
            torch.backends.cudnn.benchmark = False

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    def load_state_dict(self, state_dict, resume=True):
        self.model.load_state_dict(state_dict['model_state_dict'])

        if resume:
            hparams_dict = state_dict['hparams']
            assert hparams_dict == self.hparams.as_dict(), 'Hparams do not match! Cannot resume training.'

            self.step = state_dict['step']
            self.epoch = state_dict['epoch']
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.info(f"Resuming from Step: {self.step}, Epoch: {self.epoch}")
        else:
            # Resetting those just to be safe!!
            self._optimizer = None
            self._scheduler = None
            self.info(f"Loaded only model from state dict. Starting with Step: {self.step}, Epoch: {self.epoch}")

    def state_dict(self):
        return {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hparams': self.hparams.as_dict()
        }
    
    @staticmethod
    def checkpoint_path(checkpoint_dir, step):
        return checkpoint_dir / f'checkpoint_{step:06d}.pth'

    def _save_checkpoint(self):
        if self.disable_checkpointing:
            return
        checkpoint_path = self.checkpoint_path(self.checkpoint_dir, self.step)
        state_dict = self.state_dict()
        torch.save(state_dict, checkpoint_path)
        self.debug(f'Checkpoint saved for step: {self.step} at: {checkpoint_path}')


    @staticmethod
    def get_latest_checkpoint(checkpoint_dir):
        checkpoint_files = list(checkpoint_dir.glob('checkpoint_*.pth'))
        if len(checkpoint_files) == 0:
            return None
        checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoint_files[-1]


    def _at_training_start(self):
        if not self.disable_checkpointing:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.info(f"Training batches: {len(self.train_dl)}")
        self.info(f"Evaluation batches: {len(self.eval_dl)}")
        self.info(f'Evaluation Interval (Steps): {self.eval_interval}')

        self.debug(f'Setting torch float32 matmul precision to high for faster training!')
        torch.set_float32_matmul_precision('high')
        self._seed_everything()
        self._device = self._init_device(self._device)
        self.logger.info(f'Using device: {self._device}')
        self.model.to(self.device)
        self.model.train()

    def _log_metrics(self, suffix, metrics):
        if self.disable_checkpointing:
            return
        for k, v in metrics.items():
            self.writer.add_scalar(f'{k}/{suffix}', v, self.step)

    def _log_hparams(self, metrics={}):
        if self.disable_checkpointing:
            return
        
        if self.hparams is not None:
            hparams = {k: v for k, v in self.hparams.as_dict().items()}
            self.writer.add_hparams(hparams, metrics, run_name='.', global_step=self.step)

        self.writer.flush()

    def _at_epoch_start(self):
        self.model.train()
        self.train_metrics.reset()

    def pre_train_step(self, batch):
        pass

    def train_step(self, batch):
        raise NotImplementedError('model_step must be implemented')
    
    def post_train_step(self, batch):
        pass
    
    def _train_step(self, batch):
        # move the batch to the device
        self.pre_train_step(batch)
        batch = map_to_tensors(batch, lambda x: x.to(self.device, non_blocking=True) if x.device.type != self.device.type else x)

        self.optimizer.zero_grad()
        with torch.autocast(device_type= 'cpu' if self.device.type == 'mps' else self.device.type, dtype=torch.bfloat16):
            loss = self.train_step(batch)
        self.post_train_step(batch)

        loss.backward()
        if self.clip_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.train_metrics.add_metric('GradientNorm', norm.item())

        self.optimizer.step()
        self.scheduler.step()
        
        for idx, last_lr in enumerate(self.scheduler.get_last_lr()):
            self.train_metrics.add_metric(f'LR/ParamGroup_{idx}', last_lr)

        step_metrics = self.train_metrics.last_metrics()
        self.info(self._metrics_string("(TRAIN-STEP) ", step_metrics))
        self._log_metrics(suffix='step_train', metrics=step_metrics)

    def _at_epoch_end(self):
        epoch_metrics = self.train_metrics.mean_metrics()
        self.info(self._metrics_string("(TRAIN-EPOCH)", epoch_metrics))
        self._log_metrics(suffix='epoch_train', metrics=epoch_metrics)


    def _metrics_string(self, prefix, metrics, eval=False):

        if eval:
            num_batches = len(self.eval_dl)
            epoch_step  = self.eval_epoch_step
        else:
            num_batches = len(self.train_dl)
            epoch_step  = self.epoch_step

        epoch_progress = f'{(epoch_step*100/num_batches):6.2f}%'
        text = prefix + f" S: {self.step:4d} | Epoch: {self.epoch:2d} ({epoch_progress})" 

        skip_prefix = ['LR/ParamGroup', 'GradientNorm']
        for k, v in metrics.items():
            if any([k.startswith(p) for p in skip_prefix]):
                continue
            text += f' | {k}: {v:7.3f}'
        return text
    
    def pre_eval_step(self, batch):
        pass

    def eval_step(self, batch):
        raise NotImplementedError('eval_step must be implemented')
    
    def post_eval_step(self, batch):
        pass  
    
    def _eval_loop(self, save_checkpoint=True):
        self.model.eval()
        self.eval_metrics.reset()
        self.eval_epoch_step = 0
        for epoch_step, batch in enumerate(self.eval_dl):
            self.eval_epoch_step = epoch_step
            self.pre_eval_step(batch)
            batch = map_to_tensors(batch, lambda x: x.to(self.device) if x.device.type != self.device.type else x)
            with torch.no_grad():
                with torch.autocast(device_type= 'cpu' if self.device.type == 'mps' else self.device.type, dtype=torch.bfloat16):
                    self.eval_step(batch)
            
            self.post_eval_step(batch)
            step_metrics = self.eval_metrics.last_metrics()
            self.info(self._metrics_string(" (EVAL-STEP) ", step_metrics, eval=True))

        epoch_metrics = self.eval_metrics.mean_metrics()
        self.info(self._metrics_string("(EVAL-EPOCH) ", epoch_metrics, eval=True))
        self._log_metrics(suffix='epoch_eval', metrics=epoch_metrics)
        self._log_hparams(metrics=epoch_metrics)
        self.model.train()
        if save_checkpoint:
            self._save_checkpoint()



    def _at_training_end(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def find_lr(self, only_param_group: Optional[int] = None):
        assert isinstance(only_param_group, int) or only_param_group is None, 'only_param_group must be an integer or None'

        self.disable_checkpointing = True
        self._at_training_start()

        import matplotlib.pyplot as plt
        self.info('Running Learning Rate Finder!')
        self.info('This will plot loss vs learning for each parameter group in the optimizer')
        instructions = [
        "- Look for the learning rate where the loss starts to decrease and note where the loss stops decreasing or starts to increase rapidly. This is often considered the \"optimal\" range.",
        "- Choose a learning rate slightly below this point (often an order of magnitude below the minimum loss) as your starting learning rate."
        ]

        self.info('\n'.join(instructions))

        # Assuming `optimizer` and `train_dl` are defined
        optimizer = self.optimizer

        lr_lamdbdas = []

        total_steps = math.ceil(20*math.log10(1e7))
        for pg_idx in range(len(optimizer.param_groups)):
            lr_lambda = lambda step: 1e-7 * (10 ** (step / 20))
            if only_param_group is not None and pg_idx != only_param_group:
                lr_lambda = lambda _: 1.0
            lr_lamdbdas.append(lr_lambda)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lamdbdas)

        losses = []
        lrs = [[] for _ in range(len(optimizer.param_groups))]  # List to hold learning rates for each group

        progress = tqdm(total=total_steps, desc='Finding LR', leave=False)
        while True:
            for bidx, batch in enumerate(self.train_dl):
                batch = map_to_tensors(batch, lambda x: x.to(self.device) if x.device.type != self.device.type else x)
                optimizer.zero_grad()

                with torch.autocast(device_type= 'cpu' if self.device.type == 'mps' else self.device.type, dtype=torch.bfloat16):
                    # Forward pass to compute the loss
                    loss = self.train_step(batch)

                # Backward pass to compute the gradients
                loss.backward()
                # Update weights
                optimizer.step()
                # Step the learning rate scheduler
                scheduler.step()
                progress.update(1)
                
                # Record the loss
                losses.append(loss.item())
                # Record the current learning rates for all parameter groups
                for i, lr in enumerate(scheduler.get_last_lr()):
                    lrs[i].append(lr)
                
                    # Check the first parameter group as an example, adjust if necessary
                if all(lr > 1 for lr in scheduler.get_last_lr()):  # Stop if the LR gets too high
                    break

            if all(lr > 1 for lr in scheduler.get_last_lr()):  # Stop if the LR gets too high
                break
            

        # Plotting
        num_groups = len(optimizer.param_groups)
        if num_groups == 1:
            fig, axes = plt.subplots(num_groups, figsize=(10, 5))
            axes = [axes]  # Make it iterable
        else:
            fig, axes = plt.subplots(num_groups, 1, figsize=(10, 5 * num_groups))  # Adjust the size as needed

        for i, ax in enumerate(axes):
            ax.plot(lrs[i], losses)
            ax.set_xscale('log')
            ax.set_xlabel('Learning rate')
            ax.set_ylabel('Loss')
            ax.set_title(f'Parameter Group {i+1}')

        plot_path = self.log_dir / 'lr_finder.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
        plt.show()



    def train(self):
        try:
            self._at_training_start()

            max_steps = len(self.train_dl) * self.num_epochs if self.num_epochs is not None else float('inf')
            eval_interval = self.eval_interval if self.eval_interval is not None else len(self.train_dl)
            self.info(f'Total training steps: {max_steps}')
                
            # Run Evaluation before training starts if step > 0 (probably due to resuming from checkpoint)
            run_eval_at_start = True if self.step > 0 else False 

            while self.step <= max_steps:
                self.epoch_step = 0
                self._at_epoch_start()
                for epoch_step, batch in enumerate(self.train_dl):
                    step = epoch_step + len(self.train_dl) * self.epoch
                    if step <= self.step:
                        continue

                    if (self.step > 0 or run_eval_at_start) and self.step % eval_interval == 0:
                        self._eval_loop(save_checkpoint=False if run_eval_at_start else True)

                    self._train_step(batch)                    
                    self.epoch_step = epoch_step

                    self.step += 1
                    if self.step >= max_steps:
                        break
                    
                self._at_epoch_end()
                self.epoch += 1

            self._eval_loop(save_checkpoint=True)
        except KeyboardInterrupt:
            self.warning('Training interrupted by user')
            self._at_training_end()


    @classmethod
    def from_checkpoint(cls, Hparams_cls,
                    checkpoint_path: Union[str, Path],
                    resume=True,
                    log_level=logging.INFO,
                    disable_checkpointing_and_logging=False,
                    parent_dir=None
                ):
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f'Checkpoint file does not exist: {checkpoint_path}'

        state_dict = torch.load(checkpoint_path, map_location='cpu')
        hparams_dict = state_dict['hparams']
        hparams = Hparams_cls.from_dict(hparams_dict)
        trainer = cls(hparams,
                    log_level=log_level,
                    disable_checkpointing_and_logging=disable_checkpointing_and_logging,
                    parent_dir=parent_dir,
                    prevent_overwrite=False
                )
        if resume:
            trainer.info(f"Resuming from checkpoint: {checkpoint_path}")
        else:
            trainer.info(f"Loading model from: {checkpoint_path}")
        trainer.load_state_dict(state_dict, resume=resume)
        return trainer        
#%%
