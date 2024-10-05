import types
import warnings
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from typing import List, Callable, Union
from .utils import get_logger

logger = get_logger()

class LambdaLRWithReduceOnPlateau(LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
                 mode='min',
                 factor=0.1,
                 patience=10,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8,
                 verbose=False):
        # Initialize lambda-based scheduling parameters
        if not isinstance(lr_lambda, (list, tuple)):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        self.lambda_epoch = -1  # Independent counter for lambda-based scheduler

        # Initialize plateau-based scheduling parameters
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.best = None
        self.eps = eps
        self.verbose = verbose
        self.min_lrs = self._format_param('min_lr', optimizer, min_lr)

        # Initialize base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._last_lr = self.base_lrs.copy()

        # Initialize comparison function for plateau detection
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)

        # Now call the base class __init__
        super().__init__(optimizer, last_epoch=-1)

    def step(self):
        """Updates the learning rate based on the lambda function."""
        self.lambda_epoch += 1  # Increment independent lambda_epoch

        # Update learning rates for each param group
        new_lrs = []
        for i, (lmbda, base_lr) in enumerate(zip(self.lr_lambdas, self.base_lrs)):
            new_lr = base_lr * lmbda(self.lambda_epoch)
            # Apply min_lr constraint
            new_lr = max(new_lr, self.min_lrs[i])
            self.optimizer.param_groups[i]['lr'] = new_lr
            new_lrs.append(new_lr)
        self._last_lr = new_lrs.copy()

    def step_metric(self, metrics):
        """Adjusts learning rate based on the validation metrics."""
        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # Ignore bad epochs during cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_base_lrs()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

            if self.verbose:
                for i, lr in enumerate(self.base_lrs):
                    logger.info('Reducing base learning rate of group {} to {:.4e}.'.format(i, lr))

    def _reduce_base_lrs(self):
        for i in range(len(self.base_lrs)):
            old_base_lr = self.base_lrs[i]
            new_base_lr = max(old_base_lr * self.factor, self.min_lrs[i])
            self.base_lrs[i] = new_base_lr

        # After reducing base_lrs, recompute the current learning rates
        self.step()

    def is_better(self, current, best):
        if best is None:
            return True
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1.0 - self.threshold
            return current < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.0
            return current > best * rel_epsilon
        else:  # mode == 'max' and threshold_mode == 'abs':
            return current > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -float('inf')

        self.best = self.mode_worse

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def get_lr(self):
        return self._last_lr.copy()

    def _format_param(self, name, optimizer, param):
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(len(optimizer.param_groups), name, len(param)))
            return list(param)
        else:
            return [param] * len(optimizer.param_groups)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`."""
        state_dict = {
            'lambda_epoch': self.lambda_epoch,
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'cooldown_counter': self.cooldown_counter,
            'mode': self.mode,
            'factor': self.factor,
            'patience': self.patience,
            'threshold': self.threshold,
            'threshold_mode': self.threshold_mode,
            'cooldown': self.cooldown,
            'eps': self.eps,
            'verbose': self.verbose,
            'min_lrs': self.min_lrs,
            '_last_lr': self._last_lr,
            'base_lrs': self.base_lrs,
            'lr_lambdas': [None] * len(self.lr_lambdas),
        }


        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the scheduler's state."""
        lr_lambdas = state_dict.pop("lr_lambdas")
        self.__dict__.update(state_dict)
         # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)
