#%%
import random
import numpy as np
from unique_names_generator import get_random_name
from src.utils import generate_loguniform_numbers


#%%


SWEEP_V1 = {
    "run": lambda : get_random_name(separator="-"),
    "bs": [16, 32, 64, 128, 256],
    "prog_dim": [16],
    "heads": [8, 16],
    "blocks": [2, 3, 5],
    "n_rec_block": [1],
    "n_rec_layer": [1, 2, 3],
    "dropout": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
    "mlr": [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001],
    "plr": [None],
    "lr_warmup": [10, 50],
    "lr_decay": [1000],
    "n_steps": [None],
    "lr_schedule": ["noam", "const"],
    "mwd": [0.0, 0.1, 0.01, 0.001, 0.0001],
    "pwd": [0.0, 0.1, 0.01, 0.001, 0.0001],
    "grok_alpha": [0.8, 0.85, 0.9, 0.95, 0.99],
    "grok_lambda": [0.1, 0.5, 1, 2, 5],
    "data_aug": [0, 1],
    "num_diff_levels": [10],
    "diff_level": [5],
    "use_aux": [True],
    "seed": [42],
    "lr_find": [False],
    "bsl": [128],
    "device": [None],
    "eval_int": [None],
    "debug": [False],
    "checkpoint": [None]
}

#%%
def generate_random_sweep_config(sweep_dict, experiment_name):
    random_config = {}


    for key, value in sweep_dict.items():
        if callable(value):
            random_config[key] = value()
        else:
            random_config[key] = random.choice(value)

    random_config["experiment"] = experiment_name
    return random_config


print(generate_random_sweep_config(SWEEP_V1, "test_sweep"))

# %%

# %%
