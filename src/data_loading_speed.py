#%%
from enum import Enum
from pathlib import Path
import pickle
import sys
import time



src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory

from src.utils import hash_string
from src.tokenizer import MODEL_INPUT, MODEL_OUTPUT, ArcTokenizer
from src.dataset import ArcExamplesDataset
from src.task import ArcTrainingDataset, Example
from collections import defaultdict

from src.task import DatasetLoader
#%%

start_time = time.time()
training_data = DatasetLoader.TRAIN_ONLY.load(
    max_height=45,
    max_width=45,
    min_test=1,
    max_test=3,
    max_train=100,
    min_train=100,
)

print(f"Time taken to augment: {(time.time() - start_time)*1000} ms")
#%%
start_time = time.time()

train_examples = training_data.train_examples
eval_examples = training_data.test_examples

tokenizer = ArcTokenizer()
tokenizer.build_program_tokenizer(train_examples)
print(f"Time taken to tokenize: {(time.time() - start_time)*1000} ms")
#%%
train_ds = ArcExamplesDataset(train_examples, tokenizer)
eval_ds = ArcExamplesDataset(eval_examples, tokenizer)
#%%
#%%
#%%

#%%

from torch.utils.data import Dataset, BatchSampler, DataLoader

start_time = time.time()

train_dl = train_ds.get_dataloader(token_count=50000,
                                    pin_memory=True,
                                    shuffle=True,
                                    num_workers=4)

print(f"Time taken to get dataloader: {(time.time() - start_time)*1000} ms")

#%%
from tqdm import tqdm
start_time = None
total_time = None
for idx, batch in tqdm(train_dl):
    if idx == 0:
        start_time = time.time()
    elif idx == len(train_dl) - 1:
        total_time = (time.time() - start_time)*1000

total_batches = len(train_dl)
print(f"Time taken to iterate over dataloader: {total_time} ms")
print(f"Time taken per batch: {total_time/total_batches} ms")
# %%
# %%

#%%
51000/1113
# %%
