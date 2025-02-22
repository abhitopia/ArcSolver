#%%
import json
import math
import sys
from box import Box
from torch import nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import wandb
import numpy as np
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from tqdm import tqdm
from .utils import add_logfile_handler, add_logging_funcs, get_git_commit_hash, get_logger, map_to_tensors, migrate_hparam_dict
from dataclasses import dataclass, field
from .lrscheduler import LambdaLRWithReduceOnPlateau

def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads


def clip_dense_grad_norm_(parameters, max_norm, norm_type=2):
    """
    Clip the norm of the gradients for all dense parameters
    """
    dense_params = [p for p in parameters if p.grad is not None and not p.grad.is_sparse]
    return torch.nn.utils.clip_grad_norm_(dense_params, max_norm, norm_type)


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
        return (n / d) if d != 0 else 0
    
    def mean(self, name):
        data = self._data[name]
        denom = sum([d for n, d in data])
        numer = sum([n for n, d in data])
        return numer / denom if denom != 0 else 0

    def reset(self):
        self._data = defaultdict(list)
        self._tags = defaultdict(set)



@dataclass
class Hparams:
    experiment: str
    run: str
    clip_grad_norm: Optional[float] = 1.0 # Clip gradient norm, None means no clipping
    accumulation_steps: int = 1 # Number of steps to accumulate gradients before stepping the optimizer
    seed: int = 1337  # Seed everything for reproducibility
    device: str = None # Device to use for training, None means automatically determine the best device
    eval_interval: Optional[int] = None # Evaluate every n steps, None means evaluate after every epoch
    num_checkpoints_to_keep: int = 3

    # Used for the purposes of tracking checkpoint
    target_metric: str ='loss' # This applies to evaluation
    target_metric_increases: bool = False # This applies to both target

    # Used to control the Learning Rate Scheduler
    plt_metric: str = 'loss' # This applies to training
    plt_metric_increases: bool =False # This applies to plt metrics
    plt_warmup: int = 0 # Number of warmup steps for the Plateau
    plateau_patience: int = 3 # Number of target_metric evaluations to wait before reducing LR
    plateau_factor: float = 0.5 # Factor by which to reduce LR
    console_metrics: Optional[List[str]] = field(default_factory=lambda: ['loss'])
    grok_alpha: Optional[float] = 0.0
    grok_lambda: Optional[float] = 0.0

    def __post_init__(self):
        assert isinstance(self.experiment, str), 'experiment must be a string'
        assert isinstance(self.run, str), 'run must be a string'
        assert self.clip_grad_norm is None or self.clip_grad_norm > 0, 'clip_grad_norm must be None or a positive float'
        assert isinstance(self.seed, int), 'seed must be an integer'
        assert self.device is None or isinstance(self.device, str), 'device must be a string or None'
        assert self.eval_interval is None or self.eval_interval > 0, 'eval_interval must be None or a positive integer'

        assert self.grok_alpha >= 0, 'grok_alpha must be zero or a positive float'
        assert self.grok_lambda >= 0, 'grok_lambda must be zero or a positive'
        assert self.grok_alpha == 0 or self.grok_lambda > 0, 'grok_lambda must be provided if grok_alpha is provided'
        assert self.grok_lambda == 0 or self.grok_alpha > 0, 'grok_alpha must be provided if grok_lambda is provideds'


        if not isinstance(self.accumulation_steps, int) or self.accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be a positive integer >= 1. "
                f"Received: {self.accumulation_steps}"
            )

        self._seed_everything()
        self._state = {}

    def _seed_everything(self):
        # logging.info(f'Seeding everything with seed: {self.seed}')
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

    def build_state(self):
        raise NotImplementedError('build_state method must be implemented')

    def add_params(self, prefix='', **kwargs):
        if len(prefix) > 0:
            if not hasattr(self, prefix):
                setattr(self, prefix, Box())

        for k, v in kwargs.items():
            assert v is None or isinstance(v, (int, float, str)), f'Value of {k} must be int, float or str'
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
    
    def init_optimizer_and_lr_schedule(self, model)-> Tuple[optim.Optimizer, Union[Callable[[int], float], List[Callable[[int], float]]]]:
        raise NotImplementedError('init_optimizer method must be implemented')
    
    def init_scheduler(self, optimizer, schedule)-> LambdaLRWithReduceOnPlateau:
        scheduler = LambdaLRWithReduceOnPlateau(
            optimizer,
            lr_lambda=schedule,
            mode='max' if self.plt_metric_increases else 'min',
            factor=self.plateau_factor,
            patience=self.plateau_patience,
            warmup_epochs=self.plt_warmup,
            verbose=True
        )

        # scheduler._step_count = -1 # To prevent warning because initialation makes a first call to step automatically
        return scheduler

    def as_dict(self):
        flat_dict = {} # not flat actually as wandb supports nested dicts
        for k, v in self.__dict__.items():
            if not k.startswith('_') and not callable(v) and v is None or isinstance(v, (int, float, str)):
                flat_dict[k] = v
            elif isinstance(v, Box):
                flat_dict[k] = dict(v)

        return flat_dict
    
    @classmethod
    def from_dict(cls, hparams_dict, build_state=True):
        args = {k: v for k, v in hparams_dict.items() if not isinstance(v, dict)} 

        hparams = cls(**args)
        for k, v in hparams_dict.items():
            if isinstance(v, dict):
                hparams.add_params(k, **v)

        if build_state:
            hparams.build_state()
            
        return hparams



class TrainerBase:
    def __init__(self,
                hparams: Hparams,
                parent_dir: Optional[Union[str, Path]] = None,
                disable_checkpointing_and_logging=False,
                prevent_overwrite=True,
                logger: Optional[logging.Logger] = None
            ):
        
        assert isinstance(hparams, Hparams), 'hparams must be an instance of Hparams'
        self.hparams = hparams
        self.logger = get_logger(logger)
        add_logging_funcs(self, self.logger)

        self.log_dir = self.get_log_dir(self.hparams.experiment, self.hparams.run, parent_dir=parent_dir)
        self.checkpoint_dir = self.get_checkpoint_dir(self.hparams.experiment, self.hparams.run, parent_dir=parent_dir)

        if prevent_overwrite:
            assert not self.checkpoint_dir.exists(), f'Checkpoint directory {self.checkpoint_dir} already exists.'

        add_logfile_handler(file_path=self.log_dir / f'training.log', logger=self.logger)

        self.info(f"Hparams: {json.dumps(self.hparams.as_dict(), indent=4)}")
        self.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.disable_checkpointing_and_logging = disable_checkpointing_and_logging
        if self.disable_checkpointing_and_logging:
            self.warning(f'Checkpointing and logging are disabled. No checkpoints will be saved and no logs will be written to WandB.')

        self.step = -1
        self.epoch = 0 
        self.epoch_step = 0
        
        # From Hparams (Direct)
        self.seed = self.hparams.seed
        self.clip_grad_norm = self.hparams.clip_grad_norm
        self.eval_interval = self.hparams.eval_interval


        # From Hparams (Indirect) - Lazy loading
        self._device = None
        self._model = None
        self._optimizer = None
        self._schedule = None
        self._scheduler = None
        self._train_dl = None
        self._eval_dl = None        

        self.train_metrics = MetricLogger()
        self.eval_metrics = MetricLogger()
        self.console_metrics = set(self.hparams.console_metrics)

        self.hparams.init_dataloaders()
        self._eval_at_start = False

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.hparams.experiment,
            name=self.hparams.run,
            id=self.hparams.run,
            dir=self.log_dir,
            # track hyperparameters and run metadata
            config=self.hparams.as_dict(),
            resume="allow",  # Always allow resuming because we will handle it ourselves as manually deleting runs from the webinterface makes it impossible to create a new run with the same name
            reinit=True, # Allows multiple runs from the same script one after another
            mode="disabled" if self.disable_checkpointing_and_logging else "online"
        )

        self.target_metric_over_steps = {}
        self._ema_grads = None


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
            self._optimizer, self._schedule = self.hparams.init_optimizer_and_lr_schedule(self.model)
        return self._optimizer
    
    @property
    def scheduler(self):
        if self._scheduler is None:
            self._scheduler = self.hparams.init_scheduler(self.optimizer, self._schedule)
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
        else:
            parent_dir = Path(parent_dir)
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
            # return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(_device, str):
            return torch.device(_device)
        elif isinstance(_device, torch.device):
            return _device


    def load_state_dict(self, state_dict, load_model=True, load_step=True, load_optim=True, strict=True):
        if load_model:
            self.model.load_state_dict(state_dict['model_state_dict'], strict=strict)
            self.model.to(self.device)
            self.info(f"Loaded model from state dict")

        self._eval_at_start = True
        current_commit_hash = get_git_commit_hash()
        saved_commit_hash = state_dict.get('git_commit_hash', None)
        if saved_commit_hash is not None and saved_commit_hash != current_commit_hash:
            self.warning(f'Git commit hash mismatch: Current: {current_commit_hash}, Checkpoint: {saved_commit_hash}')

        if strict:
            hparams_dict = state_dict['hparams']
            assert hparams_dict == self.hparams.as_dict(), 'Hparams do not match! Cannot resume training.'

        if load_optim:
            assert load_model, 'Cannot load optimizer without loading model'
            assert load_step, 'Cannot load optimizer without loading step'

            self.step = state_dict['step']

            src_optim_sd = self.optimizer.state_dict()
            trg_optim_sd = state_dict['optimizer_state_dict']

            # Copy over the new weight decays (This allows changing weight decay during resume)
            for spg, tpg in zip(src_optim_sd['param_groups'], trg_optim_sd['param_groups']):
                tpg['weight_decay'] = spg['weight_decay']

            # This will overwrite the `lr` for each param group.
            self.optimizer.load_state_dict(trg_optim_sd)

            # The way it works is that Schedule uses base_lrs and uses it to modify the `lr` for each param group. 
            # That means that after the next step, `lr` will be updated based on the new `base_lrs` in the schedule.
            scheduler_sd = state_dict['scheduler_state_dict']
            # We don't update the Base LRs because we want to keep the learning rates as they were when the checkpoint was saved
            # Including reduced learning rates due to ReduceOnPlateau.
            # scheduler_sd['base_lrs'] = [pg['lr'] for pg in src_optim_sd['param_groups']]

            self.scheduler.load_state_dict(scheduler_sd)
            self.info(f"Loaded Optimizer and Scheduler from state dict")

        else:
            # Resetting those just to be safe!!
            self._optimizer = None
            self._scheduler = None

            if load_step: # If we are not loading optimizer, we can still load the step
                self.step = state_dict['step']
            
        self.info(f"Continuing from Step: {self.step}")

    def state_dict(self):
        return {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hparams': self.hparams.as_dict(),
            'git_commit_hash': get_git_commit_hash()
        }
    
    @staticmethod
    def checkpoint_path(checkpoint_dir, step):
        return checkpoint_dir / f'checkpoint_{step:06d}.pth'
    

    def _save_checkpoint(self, eval_metrics):
        if self.disable_checkpointing_and_logging:
            return
        
        target_metric = self.hparams.target_metric
        target_metric_increases = self.hparams.target_metric_increases
        num_checkpoints_to_keep = self.hparams.num_checkpoints_to_keep
        
        assert target_metric in eval_metrics, f'Target metric: {target_metric} not found in eval metrics: {eval_metrics}. You must add it to the eval_metrics'

        checkpoint_metric = eval_metrics[target_metric]
        if target_metric_increases:
            checkpoint_metric_to_sort = -checkpoint_metric
        else:
            checkpoint_metric_to_sort = checkpoint_metric

        # Format the checkpoint name to include step and metric
        checkpoint_filename = f'ckt_{self.step}_{checkpoint_metric:.3f}.pth'
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the current checkpoint
        torch.save(self.state_dict(), checkpoint_path)
        self.debug(f'Checkpoint saved for step: {self.step} at: {checkpoint_path}')
        
        # Update the list of tracked checkpoints
        self.target_metric_over_steps[self.step] = checkpoint_metric_to_sort

        # Keep only the top N best checkpoints and the latest one
        if len(self.target_metric_over_steps) > num_checkpoints_to_keep:
            # Sort steps based on the metric value (ascending for the best values)
            sorted_steps = sorted(self.target_metric_over_steps, key=self.target_metric_over_steps.get)
            steps_to_remove = sorted_steps[num_checkpoints_to_keep:]
            for step in steps_to_remove:
                if step == self.step:
                    continue  # Always keep the latest checkpoint
        
                checkpoint_to_remove = next(self.checkpoint_dir.glob(f'ckt_{step}_*.pth'))
                checkpoint_to_remove.unlink()
                self.debug(f'Deleted checkpoint at step: {step} with metric: {self.target_metric_over_steps[step]}')
                self.target_metric_over_steps.pop(step)

        # Log the best checkpoint to Wandb
        best_step = min(self.target_metric_over_steps, key=self.target_metric_over_steps.get)
        best_metric = self.target_metric_over_steps[best_step]
        wandb.run.summary[f'best_{target_metric}'] = best_metric if not target_metric_increases else -best_metric
        wandb.run.summary['best_step'] = best_step

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir):
        checkpoint_files = list(checkpoint_dir.glob('ckt_*.pth'))
        if len(checkpoint_files) == 0:
            return None
        # Sort by step number which is in the name as 'step_{step_number}'
        try:
            checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
        except (IndexError, ValueError):
            raise ValueError("Checkpoint filename format is incorrect. Expected format: 'ckt_{step}_{metric}.pth'")
        return checkpoint_files[-1]

    def _at_training_start(self):
        self.info(f"Training batches: {len(self.train_dl)}")
        self.info(f"Evaluation batches: {len(self.eval_dl)}")
        self.info(f'Evaluation Interval (Steps): {self.eval_interval}')
        self.info(f'Gradient Accumulation Steps: {self.hparams.accumulation_steps}')
        num_updates_per_epoch = len(self.train_dl) / self.hparams.accumulation_steps
        self.info(f'Number of updates per epoch: {num_updates_per_epoch}')

        # Pleateau 
        self.info(f"Plateau Metric (tracks Train): {self.hparams.plt_metric}")
        self.info(f"Effective Plateau Patience (# Updates): {num_updates_per_epoch * self.hparams.plateau_patience}")
        self.info(f"Plateau Warmup (in number of updates): {self.hparams.plt_warmup}")
        self.info(f"Plateau Warmup (from epoch #): {math.ceil(self.hparams.plt_warmup / num_updates_per_epoch)}")

        self.debug(f'Setting torch float32 matmul precision to high for faster training!')
        torch.set_float32_matmul_precision('high')
        self._device = self._init_device(self._device)
        self.logger.info(f'Using device: {self._device}')
        self.model.to(self.device)
        self.model.train()
        self.at_training_start()

    def at_training_start(self):
        pass

    def _log_metrics(self, suffix, metrics):
        if self.disable_checkpointing_and_logging:
            return
        new_metrics = {f'{k}/{suffix}': v for k, v in metrics.items()} 
        wandb.log(data=new_metrics, step=self.step)

    def _at_epoch_start(self):
        self.model.train()
        self.train_metrics.reset()
        self.at_epoch_start()

    def at_epoch_start(self):
        pass

    def pre_train_step(self, batch):
        pass

    def train_step(self, batch):
        raise NotImplementedError('model_step must be implemented')
    
    def post_train_step(self, batch):
        pass

    def post_optimizer_step(self):
        pass
    
    def _train_step(self, batch, accumulation_step):
        # move the batch to the device
        self.pre_train_step(batch)
        batch = map_to_tensors(batch, lambda x: x.to(self.device, non_blocking=True) if x.device.type != self.device.type else x)

        # Zero the gradients only at the first step of the accumulation
        if accumulation_step == 0:
            self.optimizer.zero_grad()


        with torch.autocast(device_type= 'cpu' if self.device.type == 'mps' else self.device.type, dtype=torch.bfloat16):
            loss = self.train_step(batch)

        self.post_train_step(batch)

        loss_accum = loss / self.hparams.accumulation_steps
        loss_accum.backward()

 

        # Perform optimizer and scheduler step only at the last accumulation step
        if accumulation_step == self.hparams.accumulation_steps - 1:
            # Gradient Clipping: Apply before the optimizer step
            if self.clip_grad_norm is not None:
                norm = clip_dense_grad_norm_(self.model.parameters(), 1.0)
                self.train_metrics.add_metric('GradientNorm', norm.item())

            # This is super important to disable when alpha/lambda == 0, because otherwise it keeps accumulating the gradients for all program embeddings
            if self.hparams.grok_alpha is not None:
                if self.hparams.grok_lambda is not None and self.hparams.grok_lambda > 0:
                    self._ema_grads = gradfilter_ema(self.model, grads=self._ema_grads, alpha=self.hparams.grok_alpha, lamb=self.hparams.grok_lambda)

            self.optimizer.step()
            self.scheduler.step()
            self.post_optimizer_step()
                
            for idx, last_lr in enumerate(self.scheduler.get_last_lr()):
                self.train_metrics.add_metric(f'LR/ParamGroup_{idx}', last_lr)

        self.train_metrics.add_metric('Epoch', self.epoch)
        step_metrics = self.train_metrics.last_metrics()
        self.info(self._metrics_string("(TRAIN-STEP) ", step_metrics))
        self._log_metrics(suffix='step', metrics=step_metrics)

    def _at_epoch_end(self):
        epoch_metrics = self.train_metrics.mean_metrics()

        # Track for plateau if the training metric stopped improving
        if self.hparams.plt_metric in epoch_metrics:
            self.scheduler.step_metric(epoch_metrics[self.hparams.plt_metric])

        self.info(self._metrics_string("(TRAIN-EPOCH)", epoch_metrics))
        self._log_metrics(suffix='train', metrics=epoch_metrics)
        self.at_epoch_end()

    def at_epoch_end(self):
        pass

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
            if any([k.startswith(p) for p in skip_prefix]) or (k not in self.console_metrics):
                continue
            text += f' | {k}: {v:7.3f}'
        return text
    
    def pre_eval_step(self, batch):
        pass

    def eval_step(self, batch):
        raise NotImplementedError('eval_step must be implemented')
    
    def post_eval_step(self, batch):
        pass  

    def at_eval_start(self):
        pass
    
    def _eval_loop(self, save_checkpoint=True):
        self.at_eval_start()
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
        self._log_metrics(suffix='eval', metrics=epoch_metrics)
        self.model.train()
        if save_checkpoint:
            # # This prevents the scheduler from regustering metric right after loading the checkpoint
            # # typically this value can be much higher, which can then influence the learning rate reduction even if subsequent 
            # # values are increasing. Please note this this does not help if the training is resumed, as then the scheduler will
            # # retore the past best metric.
            # self.scheduler.step_metric(epoch_metrics[self.hparams.target_metric])
            self._save_checkpoint(epoch_metrics)

        self.at_eval_end()

    def at_eval_end(self):
        pass

    def _at_training_end(self):
        wandb.finish()

    def find_lr(self, only_param_group: Optional[int] = None):
        assert isinstance(only_param_group, int) or only_param_group is None, 'only_param_group must be an integer or None'

        self.disable_checkpointing_and_logging = True
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
                if any(lr > 1 for lr in scheduler.get_last_lr()):  # Stop if the LR gets too high
                    break

            if any(lr > 1 for lr in scheduler.get_last_lr()):  # Stop if the LR gets too high
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



    def train(self, max_steps: Optional[int] = None):
        try:
            self._at_training_start()

            if max_steps is None:
                max_steps = float('inf')
                
            self.info(f'Total training steps: {max_steps}')
                
            # Run Evaluation before training starts if step > 0 (probably due to resuming from checkpoint)
            eval_interval = self.eval_interval if self.eval_interval is not None else len(self.train_dl)
            run_eval_at_start = True if self.step > 0 or self._eval_at_start else False 


            self.epoch = self.step // len(self.train_dl) if self.step > 0 else 0
            self.info(f'Setting starting epoch to: {self.epoch}')


            accumulation_steps = self.hparams.accumulation_steps
            self.info(f'Accumulating gradients over {accumulation_steps} steps.')


            while self.step < max_steps:
                self.epoch_step = 0
                self._at_epoch_start()
                for epoch_step, batch in enumerate(self.train_dl):
                    step = epoch_step + len(self.train_dl) * self.epoch
                    if step <= self.step:
                        continue
                    elif step > max_steps:
                        break

                    self.step = step  # Set the step instead of incrementing it. This ensures that self._epoch_end as 

                    if run_eval_at_start:
                        self._eval_loop(save_checkpoint=False)
                        run_eval_at_start = False

                    accumulation_step = self.step % accumulation_steps
                    self._train_step(batch, accumulation_step)

                    # In case train_dl has different number of batches per epoch
                    eval_now = (epoch_step == len(self.train_dl) - 1) if self.eval_interval is None else (self.step > 0 and self.step % eval_interval == 0)
                    if eval_now:
                        self._eval_loop(save_checkpoint=True)


                    self.epoch_step = epoch_step

                self._at_epoch_end()
                self.epoch += 1


            self._eval_loop(save_checkpoint=True)
        except KeyboardInterrupt:
            self.warning('Training interrupted by user')
            self._at_training_end()
            sys.exit(0)


    def initialise_from_checkpoint(self, checkpoint_path: Union[str, Path], strict=True, load_model=True, load_step=True, load_optim=True):
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f'Checkpoint file does not exist: {checkpoint_path}'
        state_dict = torch.load(checkpoint_path, map_location=self.device.type, weights_only=False)
        self.info(f"Initialising model from checkpoint: {checkpoint_path}")
        self.load_state_dict(state_dict,
                            load_model=load_model,
                            load_step=load_step, 
                            load_optim=load_optim, 
                            strict=strict) # Prevent loading optimizer and scheduler state

    @staticmethod
    def load_hparams_dict(checkpoint_path: Union[str, Path]):
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f'Checkpoint file does not exist: {checkpoint_path}'
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        hparams_dict = state_dict['hparams']
        return hparams_dict 
#%%
