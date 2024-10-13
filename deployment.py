#%%
from src.deploy_utils import load_tasks
from src.solver import create_solver
import torch
torch.manual_seed(42)
#%%
base_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/'
# base_path = '/teamspace/studios/this_studio/ArcSolveR/'
tasks_path = base_path + 'data/arc_kaggle_data/arc-agi_evaluation_challenges.json'
solution_path = base_path + 'data/arc_kaggle_data/arc-agi_evaluation_solutions.json'
tasks = load_tasks(tasks_path, solution_path)
#%%
ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
# ckt_path = '/teamspace/studios/work-horse/ArcSolver/runs/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
solver = create_solver(ckt_path)
# solver = Solver(ckt_path=ckt_path, bs=5, patience=10)
# solver = torch.jit.script(solver)
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
solver.to(device)
solver(tasks[0], seed=1, thinking_duration=300, min_confidence=0.001)
# %%
