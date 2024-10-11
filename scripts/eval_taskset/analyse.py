#%%
import pickle
from pathlib import Path
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory

from src.task import ARC_EVAL
path = Path('/Users/abhishekaggarwal/synced_repos/ArcSolver/scripts/eval_taskset/arc_eval.pkl')

data = pickle.load(open(path, 'rb'))
# %%
data.keys()
# %%


num_test_correct = 0
num_train_correct = 0

solved_tasks = []
partial_solved_tasks = []
unsolved_tasks = []

for k, v in data.items():
    # print(k, v['test_accuracy'], v['train_accuracy'])
    test_acc = v['test_accuracy']
    train_acc = v['train_accuracy']

    if test_acc == 1:
        num_test_correct += 1
        solved_tasks.append(k)
    elif test_acc < 1 and test_acc > 0.0:
        partial_solved_tasks.append(k)
    else:
        unsolved_tasks.append(k)

    if train_acc == 1:
        num_train_correct += 1
# %%
num_test_correct/ len(data), num_train_correct/ len(data)
# %%
# %%
loader = ARC_EVAL
tasks = loader.tasks

solved_tasks = [task for task in tasks if task.id in solved_tasks]
partial_solved_tasks = [task for task in tasks if task.id in partial_solved_tasks]
unsolved_tasks = [task for task in tasks if task.id in unsolved_tasks]
#%%
tasks[0].train[0].input.size
# %%

def get_max_size(task):
    max_size = 0
    for sample in task.train:
        max_size = max(max_size, sample.input.size)
        max_size = max(max_size, sample.output.size)
    for sample in task.test:
        max_size = max(max_size, sample.input.size)
        max_size = max(max_size, sample.output.size)
    return max_size

solved_tasks = sorted(solved_tasks, key=lambda x: get_max_size(x))
partial_solved_tasks = sorted(partial_solved_tasks, key=lambda x: get_max_size(x))
unsolved_tasks = sorted(unsolved_tasks, key=lambda x: get_max_size(x))
# %%
# %%
result = {
    'solved': solved_tasks,
    'partial_solved': partial_solved_tasks,
    'unsolved': unsolved_tasks
}

output_path = Path('/Users/abhishekaggarwal/synced_repos/ArcSolver/scripts/eval_taskset/arc_eval_result.pkl')
pickle.dump(result, open(output_path, 'wb'))
# %%
