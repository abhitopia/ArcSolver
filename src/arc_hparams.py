import math
from typing import Tuple
import torch.optim as optim
import torch.nn as nn
import math
from torch.utils.data import DataLoader

from .dataset1 import ArcExamplesDataset
from .multilevel_loss import MultiLevelLoss, exp_spacing
from .repl import REPL, REPLConfig
from .task1 import TRAIN_EVAL_COLLECTION
from .tokenizer import ArcTokenizer
from .trainer import Hparams


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

        training_data = TRAIN_EVAL_COLLECTION if self.data.include_eval else TRAIN_ONLY_COLLECTION 

        training_data.augment(
                            num_train_per_prog=self.data.num_train_per_program,
                            max_test_per_prog=self.data.max_test_per_program)
        
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
                                           min_util=optim_config.batch_min_util)
        
        eval_dl = eval_ds.get_dataloader(token_count=optim_config.eval_batch_token_count,
                                         pin_memory=True,
                                         shuffle=False,
                                         min_util=optim_config.batch_min_util)
        

        self.state['tokenizer'] = tokenizer

        print("\n\nTraining Data Loader Stats:")
        train_dl.batch_sampler.stats()
        print("\n\nEvaluation Data Loader Stats:")
        eval_dl.batch_sampler.stats()

        return train_dl, eval_dl
    

    def init_model(self)-> nn.Module:
        config = REPLConfig(
            prog_vocab_size=len(self.state['tokenizer'].program_tokenizer),
            n_dim=self.model.n_dim,
            n_embd=self.model.n_embd, 
            n_head=self.model.n_head,
            n_layer=self.model.n_layer, 
            n_state_layer=self.model.n_state_layer,
            pnorm=self.model.pnorm, 
            dropout=self.optim.dropout
        )

        model = REPL(config)
        return model


    def init_loss_fn(self) -> nn.Module:
        loss = MultiLevelLoss(
                    pad_idx=self.state['tokenizer'].grid_tokenizer.PAD_IDX,
                    edr=self.optim.edr,
                    min_pct=self.optim.mctp)
        
        spacing = exp_spacing(self.optim.num_iters, self.optim.edr, self.optim.mctp)
        print(f"Loss Error Rate per Iteration: {spacing.tolist()}")
        return loss
    

    def init_optimizer(self, model: REPL)-> optim.Optimizer:
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