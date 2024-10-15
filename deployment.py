#%%
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
# base_path = Path('/Users/abhishekaggarwal/synced_repos/ArcSolver/')
base_path = Path(__file__).parent / 'models/v9/D512E128H16B5I3.v1/'

ckt_path = base_path / 'ckt_281000_52.168.pth'
# ckt_path = base_path / 'ckt_162000_39.205.pth'
# ckt_path = 'models/v9/D512E128H16B5I3.v1/ckt_162000_39.205.pth'

git_hash = get_git_commit_hash(7)
save_path = f"{get_git_commit_hash(7)}_{ckt_path.parent.stem}_{ckt_path.stem}.pt" if git_hash else 'test.pt'

solver = create_solver(ckt_path,
                jit=True,
                save_path=save_path)
#%%
# tasks_path = base_path / 'partial_solved_challenge.json'
# solution_path = base_path / 'partial_solved_solution.json'
# tasks_path = base_path / 'solved_challenge.json'
# solution_path = base_path / 'solved_solution.json'
tasks_path = base_path / 'unsolved_challenge.json'
solution_path = base_path / 'unsolved_solution.json'

tasks = load_tasks(tasks_path, solution_path)

# for task in tasks:
#     if task.task_id == '9110e3c5':
#         break
# else:
#     print("Task not found")

# print("Task ID: ", task.task_id)    
# print(tasks[0].task_id)
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
solver.to(device)


params = SolverParams(
    thinking=300,
    bs=20,
    patience=100,
    lr=0.01,
    lrs=1.0,
    wd=0.0,
    wu=1,
    seed=42,
    mode='vbs',
    confidence=0.000001,
    metric='L',
    strategy='Rv1'
)

# with autocast('cuda'):
solution = solver(
        task=tasks[0],
        params=params)

# %%
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
