#%%
from pathlib import Path
import json
#%%
data_dir = 'data/analysis/'
json_files = Path(data_dir).rglob('*.json')
data = {}

def dedup(tasks):
    deduped = {}
    for t in tasks:
        task_id = t['task_id']
        if task_id not in deduped:
            deduped[task_id] = t
    return list(deduped.values())

for f in sorted(json_files):
    d = json.load(f.open('r'))
    data[d['iters']] = dedup(d['tasks'])

def get_examples(tasks, key='train'):
    examples = []

    for task in tasks:
        for ex in task[key]:
            g_pred, g_score = ex['prediction']['greedy']
            b_pred, b_score = zip(*[tuple(p) for p in ex['prediction']['beam']])
            examples.append((ex['output'], g_pred, b_pred))

    return examples


def accuracy(examples, greedy=False, k=5):
    correct = 0
    for o, g, b in examples:
        if greedy and o == g:
            correct += 1
        elif not greedy and o in b[:k]:
            correct += 1
    return correct / len(examples)

# %%

# %%
import numpy as np
import matplotlib.pyplot as plt

k_values = ['Greedy', 1, 2, 3, 4, 5, 10, 20, 50]  # List of k values

accuracies_per_iter = []

# Assuming data is defined somewhere and get_examples, accuracy functions exist
for iters, tasks in data.items():
    train_examples = get_examples(tasks, 'train')
    test_examples = get_examples(tasks, 'test')

    train_accuracy_at_k = []
    test_accuracy_at_k = []

    for k in k_values:
        if k == 'Greedy':
            train_accuracy_at_k.append(accuracy(train_examples, greedy=True))
            test_accuracy_at_k.append(accuracy(test_examples, greedy=True))
        else:
            train_accuracy_at_k.append(accuracy(train_examples, greedy=False, k=k))
            test_accuracy_at_k.append(accuracy(test_examples, greedy=False, k=k))

    accuracies_per_iter.append((iters, train_accuracy_at_k, test_accuracy_at_k))

# Number of models/iterations and positions for bars
num_iters = len(accuracies_per_iter)
bar_width = 0.2  # Width of each bar
k_indices = np.arange(len(k_values))  # Create numerical indices for k_values

# Create the subplots (1 row, 2 columns)
fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(14, 6))

# Function to add value annotations on top of bars
def add_value_labels(ax, bars):
    """Attach a text label above each bar in *bars*, displaying its height."""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )

test_min_accuracy, train_min_accuracy = 1.0, 1.0
test_max_accuracy, train_max_accuracy = 0.0, 0.0

# Loop through each model/iteration and plot its bars for training accuracies
for i, (iter, train_accuracy_at_k, test_accuracy_at_k) in enumerate(accuracies_per_iter):
    # Training accuracy bars
    bars_train = ax_train.bar(k_indices + i * bar_width, train_accuracy_at_k, width=bar_width, label=f'Train {iter}')
    
    # Test accuracy bars
    bars_test = ax_test.bar(k_indices + i * bar_width, test_accuracy_at_k, width=bar_width, label=f'Test {iter}')
    
    # Update min accuracy values
    test_min_accuracy = min(test_min_accuracy, min(test_accuracy_at_k))
    train_min_accuracy = min(train_min_accuracy, min(train_accuracy_at_k))

    # Update max accuracy values
    test_max_accuracy = max(test_max_accuracy, max(test_accuracy_at_k))
    train_max_accuracy = max(train_max_accuracy, max(train_accuracy_at_k))

    # Add value labels above bars
    add_value_labels(ax_train, bars_train)
    add_value_labels(ax_test, bars_test)

# Set x-ticks and labels for the training subplot
ax_train.set_xlabel('Top-k')
ax_train.set_ylabel('Accuracy')
ax_train.set_title('Training Accuracy vs Top-k Responses')
ax_train.set_xticks(k_indices + bar_width * (num_iters - 1) / 2)
ax_train.set_xticklabels(k_values)
ax_train.legend(loc='upper left')
ax_train.grid(True, axis='y')
ax_train.set_ylim([train_min_accuracy - 0.1, train_max_accuracy + 0.1])

# Set x-ticks and labels for the test subplot
ax_test.set_xlabel('Top-k')
ax_test.set_ylabel('Accuracy')
ax_test.set_title('Test Accuracy vs Top-k Responses')
ax_test.set_xticks(k_indices + bar_width * (num_iters - 1) / 2)
ax_test.set_xticklabels(k_values)
ax_test.legend(loc='upper left')
ax_test.grid(True, axis='y')
ax_test.set_ylim([test_min_accuracy - 0.1, test_max_accuracy + 0.1])

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# %%
