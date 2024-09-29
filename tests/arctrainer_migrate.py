#%%
from bisect import bisect_right
from collections import defaultdict
import sys



src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory
#%%

from src.arc_trainer1 import ArcTrainer
import random
from src.tokenizer import ArcTokenizer
from src.repl import REPL, REPLConfig
import time
import pandas as pd
import wandb
import numpy as np
import torch
from src.trainer import TrainerBase
from src.arc_hparams import ArcHparams

#%%


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
    "n_state_layer": 3,
    "num_iters": 8,
}

optimizer_config = {
    # Batch Size
    "train_batch_token_count": 10000,
    "eval_batch_token_count": 10000,
    "batch_min_util": 0.7,

    # Regularization / Weight Decay
    "dropout": 0.0,
    "wd_model": 0.1,
    "wd_prog": 0.1,
    "l1_prog": 0.0,

    # Loss
    "edr": 2,
    "mctp": 0.4,

    # Learning Rate
    "lr_model": 0.001,
    "lr_prog": 0.001,
    "lr_schedule": "noam",
    "lr_warmup_steps": 2_000,
    "lr_decay_steps": 30_000,  

    # Misc
    "clear_cache_interval": 100,
}

hparams.add_params(prefix="data", **data_config)
hparams.add_params(prefix="optim", **optimizer_config)
hparams.add_params(prefix="model", **model_config)


trainer = ArcTrainer(
    hparams=hparams,
    parent_dir='/tmp/arc_solver',
    prevent_overwrite=True,
    num_checkpoints_to_keep=4,
    disable_checkpointing_and_logging=True
)

# trainer_sd = trainer.state_dict()
# trainer.load_state_dict(trainer_sd, load_model=True, load_optim=True, strict=True)

trainer.at_training_start()
train_dl, eval_dl = trainer.train_dl, trainer.eval_dl


batch = next(iter(train_dl))
x, y = batch

for i in range(10):
    trainer.pre_train_step(batch)
    trainer.train_step(batch)
    trainer.post_train_step(batch)
# print(x.meta)
#%%
# print(trainer.state_dict())
# print(trainer.model.pte[0].weight.shape)
# trainer.post_optimizer_step()
# # %%
