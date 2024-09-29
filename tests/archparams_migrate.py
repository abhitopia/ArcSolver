#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory


from typing import Tuple
import torch.nn as nn
import torch.optim as optim
from src.arc_trainer import const_schedule, get_alt_schedulers, lin_decay_schedule, noam_schedule
from src.multilevel_loss import MultiLevelLoss, exp_spacing
from torch.utils.data import DataLoader
from src.trainer import Hparams
from src.repl import REPL, REPLConfig
from src.task import TRAIN_ONLY_COLLECTION, TRAIN_EVAL_COLLECTION
from src.dataset import ArcExamplesDataset
from src.tokenizer import ArcTokenizer

#%%
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

# %%

experiment = 'development'
run = 'this_run'
seed = 42
device = None
eval_int = 100
grok_alpha = 0.5
grok_lambda = 0.5

hparams = ArcHparams(experiment=experiment,
                    run=run, 
                    seed=seed, 
                    device=device, 
                    eval_interval=eval_int,
                    grok_alpha=grok_alpha,
                    grok_lambda=grok_lambda)

data_config = {
    # "data_aug": data_aug,
    # "diff_level": diff_level,
    # "num_diff_levels": num_diff_levels,
    # "use_aux": use_aux
    'include_eval': False,
    'num_train_per_program': 10,
    'max_test_per_program': 3,
}


model_config = {
    "n_dim": 128,
    "n_embd": 16,
    "n_head": 4,
    "n_layer": 3,
    "pnorm": 1.0,
    "dropout": 0.0,
    "n_state_layer": 3
}

optimizer_config = {
    # Batch Size
    "train_batch_token_count": 10000,
    "eval_batch_token_count": 2000,
    "batch_min_util": 0.7,

    # Regularization / Weight Decay
    "dropout": 0.0,
    "wd_model": 0.1,
    "wd_prog": 0.1,
    "l1_prog": 0.0,

    # Loss
    "num_iters": 8,
    "edr": 2,
    "mctp": 0.4,

    # Learning Rate
    "lr_model": 0.001,
    "lr_prog": 0.001,
    "lr_schedule": "noam",
    "lr_warmup_steps": 2_000,
    "lr_decay_steps": 30_000,  
}

hparams.add_params(prefix="data", **data_config)
hparams.add_params(prefix="optim", **optimizer_config)
hparams.add_params(prefix="model", **model_config)


train_dl, eval_dl = hparams.init_dataloaders()
model = hparams.init_model()
loss = hparams.init_loss_fn()
optimiser = hparams.init_optimizer(model)
scheduler = hparams.init_scheduler(optimiser)
# %%
batch = next(iter(train_dl))
# %%

x, y = batch

y.target_grid[:, 2]
# %%
out, _ = model(x, y)
loss(out, y.target_grid)
# %%
len(complexities)
# %%
## Plot the distribution of complexities
import matplotlib.pyplot as plt
import numpy as np

plt.hist(complexities, bins=50)
plt.show()
# %%
# %%
