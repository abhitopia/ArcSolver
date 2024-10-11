#%%
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
from src.repl import REPLConfig, REPL
from src.task import ARC_TRAIN, ARC_EVAL
from src.tokenizer import ArcTokenizer
from src.dataset import ArcExamplesDataset
from src.utils import map_to_tensors
from tqdm import tqdm

ckt_path = '/teamspace/studios/work-horse/ArcSolver/runs/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
# ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_162000_39.205.pth'
#%%
## Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data = torch.load(ckt_path, map_location='cpu', weights_only=False)
model_config = REPLConfig.from_dict(data['model_config'])
model = REPL(model_config)
model.load_state_dict(data['model_state_dict'], strict=True)
tokenizer = ArcTokenizer.from_dict(data['tokenizer'])
model = torch.jit.script(model)
model.to(device)
model.eval()
#%%
def get_predicted_tokens(model, tokenizer, example, device):
    pad_idx = tokenizer.grid_tokenizer.PAD_IDX
    start_idx = tokenizer.grid_tokenizer.BOS_IDX
    batch =  ArcExamplesDataset.collate_fn([example], 
                                pad_idx=pad_idx, 
                                tokenizer=tokenizer,
                                permute=False, 
                                keep_meta=False,
                                device=torch.device('cpu'))

    x, y = map_to_tensors(batch, lambda x: x.to(device, non_blocking=True))
    iter_logits, _ = model(x, y)
    logits = iter_logits
    # logits = iter_logits[-1]
    _, predicted_tokens = torch.max(logits, dim=2)
    predicted_tokens = [start_idx] + predicted_tokens[0].tolist()[:-1]
    return predicted_tokens, y.grid.tolist()[0]


loader = ARC_EVAL
tasks = loader.tasks

output_path = Path('arc_eval.pkl')

if output_path.exists():
    eval_data = pickle.load(open(output_path, 'rb'))
else:
    eval_data = {}

for task in tqdm(tasks):

    if task.id in eval_data:
        print(f"Skipping {task.id}")
        continue

    result = {
        'prog_id': task.prog_id,
        'dataset': task.dataset,
        'train': [],
        'test': []
    }

    for example in task.train:
        predicted_tokens, target_tokens = get_predicted_tokens(model, tokenizer, example, device)
        is_match = predicted_tokens == target_tokens
        result['train'].append(is_match)

    for example in task.test:
        predicted_tokens, target_tokens = get_predicted_tokens(model, tokenizer, example, device)
        is_match = predicted_tokens == target_tokens
        result['test'].append(is_match)

    result['train_accuracy'] = sum(result['train']) / len(result['train'])
    result['test_accuracy'] = sum(result['test']) / len(result['test'])

    eval_data[task.id] = result

    pickle.dump(eval_data, open(output_path, 'wb'))



# task = tasks[task_id]
# train_examples = task.train
# test_examples = task.test
# example_id = 0
# example = test_examples[example_id]


# predicted_tokens, target_tokens = get_predicted_tokens(model, tokenizer, example)
# assert target_tokens == predicted_tokens
# %%
