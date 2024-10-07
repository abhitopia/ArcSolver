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
from src.attention_vis import patch_model, visualize_attention
# %%
## Load model
data = torch.load(ckt_path, map_location='cpu', weights_only=False)
model_config = REPLConfig.from_dict(data['model_config'])
model = REPL(model_config)
model.load_state_dict(data['model_state_dict'], strict=True)
patch_model(model)
## Load Tokenizer
tokenizer = ArcTokenizer.from_dict(data['tokenizer'])
# %%
## Load Examples
loader = ARC_TRAIN
train_examples = loader.train
dataset = ArcExamplesDataset(train_examples, tokenizer)
exampled_idx = 300
print(dataset[exampled_idx])
x, y = dataset.tokenized_example(exampled_idx)
# %%
# Get attention probs
out, (attn_probs, _, _) = model(x, y, num_iters=4, return_cache=True)
# %%
prefix_len = 3
inp_len = x.grid.size(1)
out_len = y.grid.size(1)
attn_prob = attn_probs[0][0] # First Iter, First Block
assert attn_prob.size(2) == prefix_len + inp_len + out_len, "Mismatch in attention matrix size"
#%%
fig = visualize_attention(attn_probs, prefix_len+inp_len, 4)
# %%
fig.write_html(f'attention_vis_ARC_TRAIN_{exampled_idx}.html')
# %%
