#%%
from src.deploy_utils import load_tasks
from src.solver import create_solver
import torch
import warnings
# Suppress the specific RNN UserWarning
warnings.filterwarnings(
    "ignore",
    message=r".*RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning
)

torch.manual_seed(42)

ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
# ckt_path = '/teamspace/studios/work-horse/ArcSolver/runs/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
# ckt_path = '/teamspace/studios/this_studio/ArcSolveR/models/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
solver = create_solver(ckt_path,
                lr=0.01,
                jit=True,
                save_path='models/v9/D512E128H16B5I3.v1/v9.pt')
#%%
base_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/'
# base_path = '/teamspace/studios/this_studio/ArcSolveR/'
tasks_path = base_path + 'data/arc_kaggle_data/arc-agi_evaluation_challenges.json'
solution_path = base_path + 'data/arc_kaggle_data/arc-agi_evaluation_solutions.json'
# solution_path = None
tasks = load_tasks(tasks_path, solution_path)

for task in tasks:
    if task.task_id == '1a6449f1':
        break
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
solver.to(device)

solution = solver(task, 
    seed=15, 
    bs=5,
    patience=10,
    thinking=100, 
    confidence=0.001)

# %%