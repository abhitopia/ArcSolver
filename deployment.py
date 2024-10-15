#%%
from src.deploy_utils import load_tasks
from src.solver import create_solver, SolverParams
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


params = SolverParams(
    thinking=100,
    bs=15,
    patience=30,
    lr=0.01,
    wd=0.05,
    wu=10,
    seed=15,
    mode='vbs',
    confidence=0.001,
)

solution = solver(
        task=tasks[3],
        params=params)

# %%