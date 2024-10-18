#%%
from collections import defaultdict
import json
from pathlib import Path
import pickle
from typing import Dict, List, NamedTuple, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import Tensor
base_path = Path('/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1')
from box import Box

unsolved_solutions_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/unsolved_solution.json'
solved_solutions_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/solved_solution.json'


solutions = json.load(open(unsolved_solutions_path, 'r'))
solutions.update(json.load(open(solved_solutions_path, 'r')))

#%%

class TaskSolution(NamedTuple):
    task_id: str
    predictions: List[List[Tensor]]
    scores: List[List[float]]
    log: Optional[List[Dict[str, Union[float, int]]]] = None


    def to_dict(self):
        result = []
        for pred in self.predictions:
            pred1 = pred[0].tolist()
            pred2 = pred[1].tolist() if len(pred) > 0 else pred1
            result.append({'attempt_1': pred1, 'attempt_2': pred2})
        return result


result = {}

def find_k(task):
    ks = []
    for i, preds in enumerate(task.predictions):
        target = solutions[task.task_id][i]
        best_k = None
        for k, pred in enumerate(preds):
            if pred.tolist() == target:
                print("Match found", i, k + 1)
                best_k = k + 1
                break
        ks.append(best_k)

    return ks

for f in list(base_path.glob('*_solutions.pkl')):
    ds = f.stem.split('_')[0]
    bs = f.stem.split('_')[2]
    strategy = f.stem.split('_')[-2].split('.')[0]

    data = torch.load(open(f, 'rb'), weights_only=False)
    result[(ds, bs, strategy)] = []
    for task_id, task in data.items():
        assert task_id in solutions

        ks = find_k(task)

        has_solution = False
        for l in task.log:
            # print(l)
            if l['TSE'] == 1.0:
                has_solution = True
                break

        agg = {
            "task_id": task_id,
            "top_k": ks,
            "num_steps": len(task.log),
            "has_solution": has_solution
        }

        result[(ds, bs, strategy)].append(agg)        
        # for i, solution in enumerate(task):
        #     if solution is None:
        #         continue
        #     if task_id not in result:
        #         result[task_id] = {}
        #     if i not in result[task_id]:
        #         result[task_id][i] = []
        #     result[task_id][i].append(solution
    print(f.stem, ds, bs, strategy)

# %%

dt = ['solved', 'unsolved']
bss = [ '4', '8', '12', '16']
strat = ['Rv1']

df = []

for k, v in result.items():
    bs = int(k[1])
    dt = k[0]
    sum_k = 0
    count_k = 0
    solved_count = 0
    total_count = 0
    sum_steps = 0
    num_tasks = 0
    num_has_solved = 0
    for r in v:
        if r['has_solution']:
            num_has_solved += 1
        sum_steps += r['num_steps']
        num_tasks += 1
        for tk in r['top_k']:
            total_count += 1
            if tk is not None:
                sum_k += tk
                count_k += 1
                if tk <= 2:
                    solved_count += 1

    row = {
        'data': dt,
        'batch_size': bs,
        'mean_k': sum_k / count_k,
        'accuracy': solved_count/total_count,
        'mean_steps': sum_steps/num_tasks,
        'effective_mean_steps': bs * sum_steps/num_tasks,
        'mean_solved': num_has_solved/num_tasks
    }
    df.append(row)
    
    # print(k, sum_k / count_k, solved_count/total_count, sum_steps/num_tasks, bs * sum_steps/num_tasks, num_has_solved/num_tasks)

#%%
df = pd.DataFrame(df)
#%%
df
#%%
# List of outcomes to plot
outcomes = ['accuracy', 'mean_steps', 'effective_mean_steps', 'mean_solved']

# Create a plot for each outcome
for outcome in outcomes:
    plt.figure()
    for label, group in df.groupby('data'):
        group = group.sort_values('batch_size')
        plt.plot(group['batch_size'], group[outcome], marker='o', linestyle='-', label=label)
    plt.xlabel('Batch Size')
    plt.ylabel(outcome.replace('_', ' ').title())
    plt.title(f'{outcome.replace("_", " ").title()} vs Batch Size')
    plt.legend(title='Data')
    plt.grid(True)
    plt.show()
#%%
# for d in dt:
#     for bs in bss:
#         for s in strat:
#             print(d, bs, s)
            
# ## Analyse the predicted rank of the solutions
# for pred in task.predictions:
#     print(pred[0].shape, pred[1].shape)


## Analyse whether the solution was solved or not

# %%
result.keys()
# %%
('unsolved', '4', 'Rv1') 3.3333333333333335 0.09090909090909091 246.0 984.0
('unsolved', '8', 'Rv1') 1.0 0.09090909090909091 247.0 1976.0
('unsolved', '12', 'Rv1') 1.0 0.09090909090909091 227.1 2725.2
('unsolved', '16', 'Rv1') 1.0 0.09090909090909091 227.0 3632.0

('solved', '4', 'Rv1') 1.3 0.8181818181818182 258.4 1033.6
('solved', '8', 'Rv1') 1.5 0.8181818181818182 237.7 1901.6
('solved', '12', 'Rv1') 1.4 0.8181818181818182 242.2 2906.4
('solved', '16', 'Rv1') 1.4545454545454546 0.8181818181818182 238.5 3816.0