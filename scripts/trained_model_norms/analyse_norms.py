#%%
import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
#%%
import torch

from src.dataset import GridTokenizer, ProgramTokenizer
from src.archived.interpreter import Interpreter, InterpreterConfig
#%%

def load_checkpoint(checkpoint_path: str, ref=False):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    prog_tokenizer = ProgramTokenizer.from_dict(state_dict['tokenizers']['program_tokenizer'])
    grid_tokenizer = GridTokenizer.from_dict(state_dict['tokenizers']['grid_tokenizer'])
    model_config = InterpreterConfig.from_dict(state_dict['model_config'])
    checkpoint_model = Interpreter(model_config, prog_tokenizer, grid_tokenizer)
    checkpoint_model.load_state_dict(state_dict['model_state_dict'])
    return checkpoint_model
# %%

model_256 = '/Users/abhishekaggarwal/synced_repos/ArcSolver/runs/V5_11Sept/A2D5M256H8B4L8_v15/checkpoint_420211.pth'
model_512 = '/Users/abhishekaggarwal/synced_repos/ArcSolver/lightning_runs/V6_18Sept/A2D5M512H16B5L8.v2/checkpoints/checkpoint_172031.pth'

paths = [model_256, model_512]

models = {}
for path in paths:
    model = load_checkpoint(path)
    models[model.n_dim] = model
    print(model.n_dim)

# %%
model = models[256]
model_sd = model.state_dict()
for key in model_sd.keys():
    print(f"Key: {key} - Norm: {model_sd[key].norm(p=2)}")

# %%
model = models[512]
model_sd = model.state_dict()
for key in model_sd.keys():
    print(f"Key: {key} - Norm: {model_sd[key].norm(p=2)}")
# %%
