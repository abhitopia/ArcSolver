#%%
# ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v8/D512E64H8L4I4.ftw6/ckt_240000_35.545.pth'
ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v8/D256E64H8L4I4PN.v1/ckt_188000_37.940.pth'
#%%
from torch import nn
from torch.nn import functional as F
import torch
from src.tokenizer import ArcTokenizer
from src.repl import REPLConfig, REPL
from src.task import ARC_SYNTH, ARC_TRAIN, ARC_EVAL
from src.dataset import ArcExamplesDataset
# %%
## Load model
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
model.get_optimizer(0.0, 0.1, 0.1, 0.1)

# Take 2-Norm of each embedding

torch.norm(model.ate[0].weight, dim=1, p=2)
# %%
torch.norm(model.type_emb.weight, dim=1, p=2)


# %%
model.pte[0].weight.size(), model.pte[1].weight.size()
# %%
## Multiple pte[0] with pte[1]
torch.norm(model.pte[0].weight @ model.pte[1].weight.T, dim=1, p=2)

# %%
torch.norm(model.ate[0].weight @ model.ate[1].weight.T, dim=1, p=2).mean()

# %%
torch.norm(model.cte[0].weight @ model.cte[1].weight.T, dim=1, p=2).mean()
# %%
(model.pte[0].weight @ model.pte[1].weight.T).norm(dim=1, p=2).mean()
# %%
