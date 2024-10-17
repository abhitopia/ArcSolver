#%%
from pathlib import Path
from src.deploy_utils import load_tasks, train_token_count, get_batch_size
from src.solver import create_solver


base_path = Path(__file__).parent / 'models/v9/D512E128H16B5I3.v1/'
# ckt_path = base_path / 'ckt_281000_52.168.pth'

# solver = create_solver(ckt_path,
#                 jit=True,
#                 save_path=None)

# tasks_path = base_path / 'partial_solved_challenge.json'
# solution_path = base_path / 'partial_solved_solution.json'
# tasks_path = base_path / 'solved_challenge.json'
# solution_path = base_path / 'solved_solution.json'
# tasks_path = base_path / 'unsolved_challenge.json'
# solution_path = base_path / 'unsolved_solution.json'

tasks_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/data/arc_kaggle_data/arc-agi_evaluation_challenges.json'
solution_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/data/arc_kaggle_data/arc-agi_evaluation_solutions.json'

tasks = load_tasks(tasks_path, solution_path)
# %%
from matplotlib import pyplot as plt
import numpy as np
TRY_HARD = False
BTC = 16000
# BTC = 8000
min_bs = 4
max_bs = 16

bss = []
ttc = []
for task in tasks:
    tmax_bs = min(2*len(task.train), max_bs) if TRY_HARD else max_bs
    tmin_bs = max(len(task.train), min_bs) if TRY_HARD else min_bs
    bss.append(get_batch_size(task, BTC, tmin_bs, tmax_bs))
    # bs = min(int(round(60/(np.log(train_token_count(task))))), 16)
    # bs = max(bs, 0)
    # bss.append(bs)
    ttc.append(train_token_count(task))



weights = np.ones_like(bss)/float(len(bss))

plt.hist(bss, weights=weights, bins=20)
plt.show()
plt.plot(ttc, bss, 'o')
# %%
np.mean(ttc)
# %%
