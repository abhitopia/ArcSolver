#%%
import math
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
from typing import Optional, Union
from collections import defaultdict
from tqdm import tqdm
from .utils import add_logger, map_to_tensors


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


class TrainerBase:
    def __init__(self,
                experiment_name: str,
                run_name: str,
                num_epochs,
                model: nn.Module,
                optimizer: optim.Optimizer,
                train_dl: DataLoader,
                eval_dl: DataLoader,
                eval_interval=None,
                device=None,
                log_level=logging.INFO,
                seed=1337,
                log_dir: Optional[Union[str, Path]] = None,
                hparams=None,
                clip_grad_norm: Optional[float] = 1.0,
                disable_checkpointing_and_logging=False
                ):
        

        # Set up logging
        if log_dir is None:
            import __main__
            calling_script = Path(__main__.__file__)
            log_dir = calling_script.parent / 'runs'


        assert experiment_name is not None, 'experiment_name must be provided'

        self.exp_name = experiment_name
        self.run_name = run_name
        assert self.run_name is not None, 'run_name must be provided'

        self.log_dir = Path(log_dir) / self.exp_name / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        add_logger(obj=self,
                log_level=log_level,
                name=self.run_name,
                file_path=self.log_dir / f'training.log')

        self.info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.disable_checkpointing = disable_checkpointing_and_logging
        if self.disable_checkpointing:
            self.warning(f'It is a trial run. No checkpoints or Tensorboard summaries will be saved!')
            self.warning(f'Resuming from checkpoints still works!')

        self.writer = None
        self.step = 0
        self.epoch = 0 
        self.epoch_step = 0
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.log_level = log_level
        self.train_dl = train_dl
        self.eval_dl = eval_dl
        self.eval_interval = eval_interval
        self.hparams = hparams

        if self.eval_interval is None:
            self.eval_interval = len(self.train_dl)

        self.train_metrics = MetricLogger()
        self.eval_metrics = MetricLogger()
        self.seed = seed
        self._device = device
        self.clip_grad_norm = clip_grad_norm
        self.scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.multiplicative_lr_factor_schedule)

    @property
    def device(self):
        if self._device is None:
            self._device = self._init_device(self._device)
        return self._device

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


    def _load_checkpoint(self, checkpoint_file):
        state_dict = torch.load(checkpoint_file, map_location=self.device)
        self.load_state_dict(state_dict)
        self.info(f"Loaded checkpoint from: {checkpoint_file}")

    def load_state_dict(self, state_dict):
        hparams = state_dict['hparams']
        assert hparams == self.hparams, 'Hparams do not match! Cannot resume training.'
        self.step = state_dict['step']
        self.epoch = state_dict['epoch']
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])


    def _resume(self):
        # Check if there is a checkpoint to resume from
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_*.pth'))
        if len(checkpoint_files) > 0:
            checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            checkpoint_file = checkpoint_files[-1]
            self._load_checkpoint(checkpoint_file)
            self.info(f'Resuming training from step {self.step} and epoch {self.epoch}')
        else:
            self.model.to(self.device)

    def _at_training_start(self):
        if not self.disable_checkpointing:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        self.info(f'Epochs: {self.num_epochs}')
        self.info(f"Training batches: {len(self.train_dl)}")
        self.info(f"Evaluation batches: {len(self.eval_dl)}")
        self.info(f'Evaluation Interval (Steps): {self.eval_interval}')

        self.debug(f'Setting torch float32 matmul precision to high for faster training!')
        torch.set_float32_matmul_precision('high')
        self._seed_everything()
        self._device = self._init_device(self._device)
        self.logger.info(f'Using device: {self._device}')
        self._resume()
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
            hparams = {k: v for k, v in self.hparams.items()}
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

        loss.backward()

        if self.clip_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.train_metrics.add_metric('GradientNorm', norm.item())

        self.optimizer.step()

        for idx, last_lr in enumerate(self.scheduler.get_last_lr()):
            self.train_metrics.add_metric(f'LR/ParamGroup_{idx}', last_lr)

        self.scheduler.step()

        self.post_train_step(batch)
        step_metrics = self.train_metrics.last_metrics()
        self.info(self._metrics_string("(TRAIN-STEP) ", step_metrics))
        self._log_metrics(suffix='step_train', metrics=step_metrics)

    def _at_epoch_end(self):
        epoch_metrics = self.train_metrics.mean_metrics()
        self.info(self._metrics_string("(TRAIN-EPOCH)", epoch_metrics))
        self._log_metrics(suffix='epoch_train', metrics=epoch_metrics)


    def _metrics_string(self, prefix, metrics):
        num_batches = len(self.train_dl)
        epoch_progress = f'{(self.epoch_step*100/num_batches):6.2f}%'
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
        for batch in self.eval_dl:
            self.pre_eval_step(batch)
            batch = map_to_tensors(batch, lambda x: x.to(self.device) if x.device.type != self.device.type else x)
            with torch.no_grad():
                with torch.autocast(device_type= 'cpu' if self.device.type == 'mps' else self.device.type, dtype=torch.bfloat16):
                    self.eval_step(batch)
                    
            self.post_eval_step(batch)

            step_metrics = self.eval_metrics.last_metrics()
            if len(step_metrics) > 0:
                self.info(self._metrics_string(" (EVAL-STEP) ", step_metrics))

        epoch_metrics = self.eval_metrics.mean_metrics()
        self.info(self._metrics_string(" (EVAL-EPOCH)", epoch_metrics))
        self._log_metrics(suffix='epoch_eval', metrics=epoch_metrics)
        self._log_hparams(metrics=epoch_metrics)
        self.model.train()
        if save_checkpoint:
            self._save_checkpoint()

    def pre_checkpoint_save(self, state_dict):
        self.debug('pre_checkpoint_save method not implemented, using default state_dict')

    def state_dict(self):
        return {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hparams': self.hparams
        }

    def _save_checkpoint(self):
        if self.disable_checkpointing:
            return
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{self.step:06d}.pth'
        state_dict = self.state_dict()
        # self.pre_checkpoint_save(state_dict)
        torch.save(state_dict, checkpoint_path)

    def multiplicative_lr_factor_schedule(self, step):
        if step == 1:
            self.warning('multiplicative_lr_factor_schedule method not implemented. Using a constant factor 1')
        return 1.0

    def _at_training_end(self):
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



    def train(self, max_steps=None):
        if max_steps is None:
            max_steps = len(self.train_dl) * self.num_epochs
        try:
            self._at_training_start()

            self.info(f'Total training steps: {max_steps}')
            # Run Evaluation before training starts
            if self.step > 0:
                # Only run evaluation if we are resuming from a checkpoint
                self._eval_loop(save_checkpoint=False)

            for epoch in range(self.num_epochs):
                if epoch < self.epoch:
                    self.debug(f'Skipping epoch {epoch}')
                    continue
                self.epoch = epoch

                self.epoch_step = 0
                self._at_epoch_start()
                for epoch_step, batch in enumerate(self.train_dl):
                    step = epoch_step + len(self.train_dl) * self.epoch
                    if step <= self.step:
                        continue
                    
                    self._train_step(batch)
                    self.step += 1
                    self.epoch_step = epoch_step

                    if self.step % self.eval_interval == 0:
                        self._eval_loop(save_checkpoint=True)

                    if self.step >= max_steps:
                        break
                    
                self._at_epoch_end()
                if self.step >= max_steps:
                    break

            self._eval_loop(save_checkpoint=True)
        except KeyboardInterrupt:
            self.warning('Training interrupted by user')
            self._at_training_end()


    @staticmethod
    def from_checkpoint(checkpoint_path):
        raise NotImplementedError('from_checkpoint method must be implemented')
#%%
