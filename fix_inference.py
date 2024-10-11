#%%
import torch
import torch.nn.functional as F
from src.repl import REPLConfig, REPL
from src.task import ARC_SYNTH, ARC_TRAIN
from src.tokenizer import ArcTokenizer
from src.dataset import ArcExamplesDataset
# ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_162000_39.205.pth'
ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
#%%
## Load model

device = 'cpu'
## Load model
data = torch.load(ckt_path, map_location='cpu', weights_only=False)
model_config = REPLConfig.from_dict(data['model_config'])
model = REPL(model_config)
model.load_state_dict(data['model_state_dict'], strict=True)
## Load Tokenizer
tokenizer = ArcTokenizer.from_dict(data['tokenizer'])
#%%

#%%

model = torch.jit.script(model)
model.eval()

def get_predicted_tokens(model, tokenizer, example):
    pad_idx = tokenizer.grid_tokenizer.PAD_IDX
    start_idx = tokenizer.grid_tokenizer.BOS_IDX
    x, y =  ArcExamplesDataset.collate_fn([example], 
                                pad_idx=pad_idx, 
                                tokenizer=tokenizer,
                                permute=False, 
                                keep_meta=False,
                                device=torch.device('cpu'))

    iter_logits, _ = model(x, y)
    logits = iter_logits
    _, predicted_tokens = torch.max(logits, dim=2)
    predicted_tokens = [start_idx] + predicted_tokens[0].tolist()[:-1]
    return predicted_tokens, y.grid.tolist()[0]

def get_greedy_prediction(model, tokenizer, example):
    x, y =  tokenizer.encode(example)
    cp = x.color_permutation[0]
    at = x.array_transform[0]
    prg = x.program[0]
    inp_grid = x.grid
    inp_indices = x.grid_indices

    gs, score = model.greedy_search(
        prog_idx=prg,
        input_grid=inp_grid,
        input_indices=inp_indices,
        color_perm_idx=cp,
        array_tform_idx=at
    )

    return gs, score

def get_beam_search_prediction(model, tokenizer, example, top_k=5, prob_thresh=0.01):
    x, y =  tokenizer.encode(example)
    cp = x.color_permutation[0]
    at = x.array_transform[0]
    prg = x.program[0]
    inp_grid = x.grid
    inp_indices = x.grid_indices
    result = model.beam_search(
        input_grid=inp_grid,
        input_indices=inp_indices,
        prog_idx=prg,
        color_perm_idx=cp,
        array_tform_idx=at,
        top_k=top_k,
        prob_thresh=prob_thresh
    )

    return result[0]

# %%
loader = ARC_TRAIN
tasks = loader.tasks
task_id = 8

task = tasks[task_id]
train_examples = task.train
test_examples = task.test
example_id = 2
example = train_examples[example_id]
# %%

predicted_tokens, target_tokens = get_predicted_tokens(model, tokenizer, example)
assert target_tokens == predicted_tokens
# %%
#%%
gs, score = get_greedy_prediction(model, tokenizer, example)
assert gs == target_tokens

#%%
bs, score_bs = get_beam_search_prediction(model, tokenizer, example, top_k=5, prob_thresh=0.01)
assert bs == target_tokens
# %%
score
# %%
