#%%
import pickle
import json
from pathlib import Path 

v9_data = '/Users/abhishekaggarwal/synced_repos/ArcSolver/scripts/eval_taskset/arc_eval_result.pkl'
output_folder = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1'
challenge_json = Path('/Users/abhishekaggarwal/synced_repos/ArcSolver/data/arc_kaggle_data/arc-agi_evaluation_challenges.json')
solution_json = Path('/Users/abhishekaggarwal/synced_repos/ArcSolver/data/arc_kaggle_data/arc-agi_evaluation_solutions.json')

with open(v9_data, 'rb') as f:
    data = pickle.load(f)

challenge_tasks = json.load(challenge_json.open('r'))
solution_tasks = json.load(solution_json.open('r'))
all_task_ids = set(challenge_tasks.keys())
# %%

for key in data.keys():
    key_task_ids = set([t.prog_id for t in data[key]])
    common_task_ids = all_task_ids.intersection(key_task_ids)
    print(f'{key}: {len(common_task_ids)}')

    challenge = {}
    for task_id in common_task_ids:
        challenge[task_id] = challenge_tasks[task_id]

    solution = {}
    for task_id in common_task_ids:
        solution[task_id] = solution_tasks[task_id]

    challenge_path = Path(output_folder) / f'{key}_challenge.json'
    with challenge_path.open('w') as f:
        json.dump(challenge, f)

    solution_path = Path(output_folder) / f'{key}_solution.json'
    with solution_path.open('w') as f:
        json.dump(solution, f)
    
#%% 
solution['696d4842']
# %%
