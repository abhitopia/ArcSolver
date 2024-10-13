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
#%%
base_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/'
# base_path = '/teamspace/studios/this_studio/ArcSolveR/'
tasks_path = base_path + 'data/arc_kaggle_data/arc-agi_evaluation_challenges.json'
# solution_path = base_path + 'data/arc_kaggle_data/arc-agi_evaluation_solutions.json'
solution_path = None
tasks = load_tasks(tasks_path, solution_path)

#%%
ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
# ckt_path = '/teamspace/studios/work-horse/ArcSolver/runs/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
solver = create_solver(ckt_path,
                lr=0.01,
                jit=True,
                save=True)

for task in tasks:
    if task.task_id == '0a1d4ef5':
        break
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
solver.to(device)
for task in tasks:
    solution = solver(task, 
        seed=15, 
        bs=5,
        patience=10,
        thinking=10, 
        confidence=0.001)
# %%
# for pred in solution.predictions:
#     pred1 = pred[0].tolist()
#     pred2 = pred[1].tolist()
#     print(pred1)
#     print(pred2)
# %%



        

