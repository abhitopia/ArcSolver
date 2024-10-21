#%%
import math
from src.deploy_utils import load_tasks
from src.solver import create_solver, SolverParams
import torch
from pathlib import Path
import warnings

from src.utils import get_git_commit_hash
# Suppress the specific RNN UserWarning
warnings.filterwarnings(
    "ignore",
    message=r".*RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning
)

#%%
torch.manual_seed(42)
# base_path = Path(__file__).parent / 'models/v9/D512E128H16B5I3.v1/'
# ckt_path = base_path / 'ckt_281000_52.168.pth'

base_path = Path(__file__).parent / 'models/v10/D384E256H12B6I4P.v2/'
ckt_path = base_path / 'ckt_692000_53.555.pth'
# base_path = Path(__file__).parent / 'models/v9/D512E128H16B5I3.ft/'
# ckt_path = base_path / 'ckt_74739_34.615.pth'

git_hash = get_git_commit_hash(7)
save_path = f"{get_git_commit_hash(7)}_{ckt_path.parent.name}_{ckt_path.stem}.pt" if git_hash else 'test.pt'

solver = create_solver(ckt_path,
                jit=True,
                save_path=save_path,
                optimize=False
                )

# solver = torch.load(save_path)
#%%
base_path = Path(__file__).parent / 'models/v9/D512E128H16B5I3.v1/'
# tasks_path = base_path / 'partial_solved_challenge.json'
# solution_path = base_path / 'partial_solved_solution.json'
tasks_path = base_path / 'solved_challenge.json'
solution_path = base_path / 'solved_solution.json'
# tasks_path = base_path / 'unsolved_challenge.json'
# solution_path = base_path / 'unsolved_solution.json'

tasks = load_tasks(tasks_path, solution_path)
#%%


# for task in tasks:
#     print(task.task_id, task.complexity())

# for task in tasks:
#     if task.task_id == 'ca8f78db':
#         break
# else:
#     print("Task not found")

# print("Task ID: ", task.task_id)    
# print(tasks[0].task_id)
task = tasks[4]
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
solver.to(device)

params = SolverParams(
    thinking=200,
    lr_factor=1.0,
    lr_patience=1,
    btc = 5000,
    min_bs = 4,
    max_bs = 4,
    patience=50,
    lr=0.05,
    lrs=1.0,
    wd=0.0,
    wu=1,
    seed=42,
    mode='vbs',
    metric='L',
    strategy='1vR',
    zero_init=False,
    predict=True,
    top_k=3,
    num_beams=9,
    return_logs=True
)

# with autocast('cuda'):
solution = solver(
        task=task,
        params=params)

# %%
# solution.log

# %%
# Solves tasks[0] of partial_solved
# params = SolverParams(
#     thinking=250,
#     bs=20,
#     patience=30,
#     lr=0.01,
#     lrs=1.0,
#     wd=0.0,
#     wu=1,
#     seed=42,
#     mode='vbs',
#     confidence=0.000001,
#     metric='L',
#     strategy='Rv1'
# )


# params = SolverParams(
#     thinking=500,
#     bs=10,
#     patience=100,
#     lr=0.01,
#     lrs=1.0,
#     wd=0.0,
#     wu=1,
#     seed=42,
#     mode='vbs',
#     confidence=0.000001,
#     metric='L',
#     strategy='Rv1'
# )

#%%