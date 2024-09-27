#%%
import math
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory


import torch
import torch.nn.functional as F
import torch.nn as nn

from src.rope_2d import Rope2D

bs = 1
dim = 16
nh = 4
h_dim = dim // nh
t = 8
d = 6

# attn_mask = torch.ones(bs, t, t).bool().tril(diagonal=0).unsqueeze(1)

# # zero out the first row of the attention mask
# attn_mask[:, :, 0, :] = False
# attn_mask[:, :, 4, :] = False


q = torch.rand(1, dim)
v = torch.rand(1, dim)
q = q.view(1, 1, dim).expand(bs, t, dim)
q = q.view(bs, t, nh, h_dim).transpose(1, 2)
v = v.view(1, 1, dim).expand(bs, t, dim)
k = q.clone()


indices = torch.tensor([[0, 0], [-1, -1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3]])
indices = indices.unsqueeze(0).expand(bs, t, 2)
q.size(), indices.size()

# %%
q[0, :, 0,  :], q[0, :, 7, :]
# %%
indices[0, 0, :], indices[0, 1, :]
# %%

rope = Rope2D(h_dim=h_dim, max_height=30, max_width=30)

q_roped = rope(q, indices)
k_roped = rope(k, indices)
q_roped.size()
# %%
# %%
q_roped[0, 0, 0, :], q_roped[0, 0, 1, :]
# %%
k_roped[0, 0, 0, :], k_roped[0, 0, 3, :]
#%%

qA = q_roped[0, :, 0, :]
qB = q_roped[0, :, 1, :]

qC = q_roped[0, :, 4, :]
qD = q_roped[0, :, 5, :]


scoreAB = torch.einsum('n d, n d -> n', qA, qB)
scoreCD = torch.einsum('n d, n d -> n', qC, qD)

# scoreAB = qA * qB
# scoreCD = qC * qD
assert torch.allclose(scoreAB, scoreCD)
# %%
scoreAB, scoreCD

# %%
qA = q_roped[0, 0, 2, :]
qB = q_roped[0, 0, 4, :]

qC = q_roped[0, 0, 3, :]
qD = q_roped[0, 0, 6, :]

(qA * qB).sum(), (qC * qD).sum()
# %%
q[0, :, 5, :], q_roped[0, :, 5, :]
# %%
