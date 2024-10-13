#%%
from src.deploy_utils import load_tasks
from src.solver import create_solver
import torch
torch.manual_seed(42)
#%%
tasks_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/data/arc_kaggle_data/arc-agi_evaluation_challenges.json'
solution_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/data/arc_kaggle_data/arc-agi_evaluation_solutions.json'
tasks = load_tasks(tasks_path, solution_path)
#%%
# ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/D512E128H16B5I3.v1_ckt_281000_52.168.pt'
ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
solver = create_solver(ckt_path)
# solver = Solver(ckt_path=ckt_path, bs=5, patience=10)
# solver = torch.jit.script(solver)
#%%
solver(tasks[0], seed=1, thinking_duration=300, min_confidence=0.001)