#%%
# ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v8/D512E64H8L4I4.ftw6/ckt_240000_35.545.pth'
#%%
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
import torch
from src.tokenizer import ArcTokenizer
from src.repl import REPLConfig, REPL
from src.task import ARC_SYNTH, ARC_TRAIN, ARC_EVAL
from src.dataset import ArcExamplesDataset
# %%
## Load model
ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'

data = torch.load(ckt_path, map_location='cpu', weights_only=False)
model_config = REPLConfig.from_dict(data['model_config'])
model_config.lora_r = 1
model = REPL(model_config)
#%%
model.load_state_dict(data['model_state_dict'], strict=False)
list(model.state_dict().keys())
model_config
# %%
for name, param in model.named_parameters():
    # Take 2-Norm of the parameters
    norm = torch.norm(param)
    print(f'{name}: {norm.item()}')

# %%
# Take 2-Norm of each embedding
embd = model.pte

emdb_weight = embd[0].weight
embd_proj_weight = embd[1].weight

projected_weight = torch.matmul( emdb_weight, embd_proj_weight.T )
print(projected_weight.shape)
plt.hist(projected_weight.norm(dim=1).detach().numpy(), bins=100)
# torch.norm(model.ate[0].weight, dim=1, p=2)
# %%
