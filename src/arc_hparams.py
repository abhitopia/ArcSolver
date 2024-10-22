from dataclasses import dataclass, field
import math
from typing import Callable, List, Optional, Tuple
import torch
import torch.optim as optim
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .dataset import ArcExamplesDataset
from .repl import REPL, REPLConfig
from .task import get_task_loaders, load_dataset
from .tokenizer import ArcTokenizer
from .trainer import Hparams
from .utils import get_logger

logger = get_logger()


def noam_schedule(step, warmup_steps, decay_steps, min_lr_scale=0.1):
    """
    Computes the learning rate at a given step based on the adjusted Noam scheduler.

    Args:
        step (int): Current training step.
        warmup_steps (int): Number of warmup steps.
        decay_steps (int): Number of steps for cosine decay.
        min_lr_scale (float): Scaling factor for the minimum learning rate.

    Returns:
        float: Learning rate at the current step.
    """
    max_lr = 1.0
    min_lr = max_lr * min_lr_scale
    step_until_decay = warmup_steps + decay_steps

    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    elif step <= step_until_decay:
        # Cosine decay
        decay_ratio = (step - warmup_steps) / (step_until_decay - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    else:
        # Inverse square root decay
        return min_lr * (step_until_decay / step) ** 0.5


def const_schedule(step, warmup_steps):
    max_lr = 1.0
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    return max_lr


def lin_decay_schedule(step, warmup_steps, decay_steps, min_lr_scale=0.1):
    max_lr = 1.0
    min_lr = max_lr * min_lr_scale
    step_until_decay = warmup_steps + decay_steps

    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if step > step_until_decay:
        return min_lr
    # 3) in between, use linear decay down to min learning rate
    decay_ratio = (step - warmup_steps) / (step_until_decay - warmup_steps)
    assert 0 <= decay_ratio <= 1
    return max_lr - decay_ratio * (max_lr - min_lr)

def get_alt_schedulers(num_steps_in_epoch):
    first_scheduler = lambda step: 1.0 if (step // num_steps_in_epoch) % 2 == 0 else 0.0
    second_scheduler = lambda step: 0.0 if (step // num_steps_in_epoch) % 2 == 0 else 1.0
    return first_scheduler, second_scheduler


@dataclass
class ArcHparams(Hparams):
    console_metrics: List[str] = field(default_factory=lambda: ['Loss', 'SampleAcc(%)', 'TokenAcc(%)', 'ΔT(ms)', '#TokensPerSec'])

    def build_state(self):
        self.reset_state()

        task_loaders = get_task_loaders(
                        train=self.data.include_train,
                        evl=self.data.include_eval,
                        aux=self.data.include_aux,
                        inv=self.data.include_inv)
        
        assert len(task_loaders) > 0, "No Task Loaders Selected"
        logger.info(f"Augmenting examples to be in range:\n Test: [{self.data.min_test_pp}, {self.data.max_test_pp}], Train:[{self.data.min_train_pp}, {self.data.max_train_pp}]")

        training_data = load_dataset(
            task_loaders=task_loaders,
            max_height=45,
            max_width=45,
            min_test=self.data.min_test_pp,
            max_test=self.data.max_test_pp,
            max_train=self.data.max_train_pp,
            min_train=self.data.min_train_pp,
        )

        training_data.stats()

        train_examples = training_data.train_examples
        eval_examples = training_data.test_examples

        tokenizer = ArcTokenizer()
        tokenizer.build_program_tokenizer(train_examples)

        train_ds = ArcExamplesDataset(train_examples, tokenizer)
        eval_ds = ArcExamplesDataset(eval_examples, tokenizer)

        optim_config = self.optim

        train_dl = train_ds.get_dataloader(token_count=optim_config.train_batch_token_count,
                                           pin_memory=True,
                                           shuffle=True,
                                           permute=self.data.permute,
                                           num_workers=4 if torch.cuda.is_available() else 0)
        
        eval_dl = eval_ds.get_dataloader(token_count=optim_config.eval_batch_token_count,
                                         pin_memory=True,
                                         shuffle=False,
                                         permute=False,
                                         num_workers=4 if torch.cuda.is_available() else 0)
        
        self.state['train_dl'] = train_dl
        self.state['eval_dl'] = eval_dl
        self.state['tokenizer'] = tokenizer
        logger.info("\n\nTraining Data Loader Stats:")
        train_dl.batch_sampler.stats()
        logger.info("\n\nEvaluation Data Loader Stats:")
        eval_dl.batch_sampler.stats()

        ## MODEL
        config = REPLConfig(
            prog_vocab_size=len(self.state['tokenizer'].program_tokenizer),
            n_dim=self.model.n_dim,
            n_embd=self.model.n_embd, 
            n_head=self.model.n_head,
            n_layer=self.model.n_layer, 
            pnorm=self.model.pnorm, 
            dropout=self.optim.dropout,
            n_iter=self.model.n_iter,
            rope_base=self.model.rbase,
            gamma=self.model.gamma,
        )

        self.state['model'] = REPL(config)
        self.state['model'].print_parameters()

        ## LOSS
        # loss = MultiLevelLoss(
        #             pad_idx=self.state['tokenizer'].grid_tokenizer.PAD_IDX,
        #             edr=self.optim.edr,
        #             min_pct=self.optim.mctp)
        
        # spacing = exp_spacing(self.optim.n_iter, self.optim.edr, self.optim.mctp)
        # logger.info(f"\nLoss Error Rate per Iteration: {[f'{c:.2f}' for c in spacing.tolist()]}")


    def init_dataloaders(self)-> Tuple[DataLoader, DataLoader]:
        if 'train_dl' not in self.state:
            self.build_state()

        train_dl = self.state['train_dl']
        eval_dl = self.state['eval_dl']
        return train_dl, eval_dl
    
    def init_model(self)-> nn.Module:
        if 'model' not in self.state:
            self.build_state()
        return self.state['model']
    

    def init_optimizer_and_lr_schedule(self, model: REPL)-> Tuple[optim.Optimizer, List[Callable[[int], float]]]:
        config = self.optim

        if config.lr_prog is None:
            config.lr_prog = config.lr_model

        optimizer = model.get_optimizer(
                                    model_lr=config.lr_model,
                                    prog_lr=config.lr_prog,
                                    model_wd=config.wd_model,
                                    prog_wd=config.wd_prog,
                                    prog_l1=config.l1_prog,
                                    device_type=self.device,
                                )
    
        config = self.optim
        warmup_steps = config.lr_warmup_steps
        decay_steps = config.lr_decay_steps
        lr_min_scale = config.lr_min_scale

        if config.lr_schedule == 'noam': 
            schedule = lambda step: noam_schedule(step, warmup_steps, decay_steps, lr_min_scale)
        elif config.lr_schedule == 'lindecay':
            schedule = lambda step: lin_decay_schedule(step, warmup_steps, decay_steps)
        elif config.lr_schedule == 'const':
            schedule = lambda step: const_schedule(step, warmup_steps)
        elif config.lr_schedule == 'alt':
            assert len(optimizer.param_groups) == 3, "Invalid LR Schedule"
            high_low_schedule, low_high_schedule = get_alt_schedulers(config.lr_decay_steps)
            schedule = [high_low_schedule, low_high_schedule, low_high_schedule]
        else:
            raise ValueError(f"Invalid LR Schedule: {config.lr_schedule}. Options: noam, const, alt")

        return optimizer, schedule