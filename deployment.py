#%%
from typing import NamedTuple
from src.deploy_utils import load_tasks
from src.solver import create_solver
import torch
from pathlib import Path
import warnings
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
# ckt_path = 'models/v9/D512E128H16B5I3.v1/ckt_162000_39.205.pth'

solver = create_solver(ckt_path,
                jit=True,
                save_path=None)
#%%
tasks_path = base_path / 'solved_challenge.json'
solution_path = base_path / 'solved_solution.json'
# solution_path = None
tasks = load_tasks(tasks_path, solution_path)

for task in tasks:
    if task.task_id == '9110e3c5':
        break
else:
    print("Task not found")

print("Task ID: ", task.task_id)    
print(tasks[0].task_id)
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
solver.to(device)

class SolverParams(NamedTuple):
    thinking: int = 500
    bs: int = 25
    patience: int = 30
    lr: float = 0.005
    wd: float = 0.05
    seed: int = 60065
    mode: str = '60065'
    confidence: float = 0.0001

params = SolverParams(
    thinking=100,
    bs=15,
    patience=30,
    lr=0.0001,
    wd=0.05,
    seed=15,
    mode='15',
    confidence=0.001,
)

solution = solver(
        task=tasks[3],
        params=params)

# %%