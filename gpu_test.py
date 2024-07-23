
#%%
import time
import math
import torch
import logging
from src.dataset import TrainingData, ProgramTokenizer, GridTokenizer
from src.interpreter import Interpreter, InterpreterConfig
from src.utils import nearest_greater_power_of_2
#%%

# Training Data Configuration


AUGMENTATION_FACTOR = 2
JOIN_VERSION = False

# Global Configuration
SEED = 42


training_data = TrainingData(augmentation_factor=AUGMENTATION_FACTOR,
                            join_version=JOIN_VERSION, 
                            seed=SEED).load()




program_tokenizer = training_data.program_tokenizer
grid_tokenizer = training_data.grid_tokenizer

PROGRAM_VOCAB_SIZE = nearest_greater_power_of_2(len(program_tokenizer))
GRID_VOCAB_SIZE = nearest_greater_power_of_2(len(grid_tokenizer))
#%%

## Model Set up
N_LAYERS = 3
N_MIXERS = 3
N_BLOCKS = 3
N_HEADS = 16
N_DIM = 128

model_config = InterpreterConfig(
    prog_vocab_size = PROGRAM_VOCAB_SIZE,
    grid_vocab_size = GRID_VOCAB_SIZE,
    n_dim = N_DIM, # dimension of the model
    n_head = N_HEADS, # number of heads within each self-attention block
    n_mixers = N_MIXERS, # number of self-attention layers within each transformer block
    n_blocks = N_BLOCKS, # number of transformer blocks within each recurrence block
    n_rec_layers = N_LAYERS # number of recurrences
)




# Training Set up
MODEL_WD = 0.01
MODEL_LR = 0.01
PROG_LR_SCALE = 10
PROG_WD_SCALE = 0.0
PROGRAM_SCALE = 0.1   # <1 So program embeddings are moved slower across batches.

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BS = 128
SEQ_LEN = 1024
DYNAMIC_BATCHING = False
PIN_MEMORY = True
USE_COMPILE = False
DATA_DEVICE = torch.device('cpu')



trains_ds = training_data.train_ds
train_dl = trains_ds.get_dataloader(batch_size=BS,
                                    seq_len=SEQ_LEN,
                                    batch_by_token_count=DYNAMIC_BATCHING,
                                    device=DATA_DEVICE,
                                    pin_memory=PIN_MEMORY)


print(len(train_dl))

# %%
torch.set_float32_matmul_precision('high')
model = Interpreter(model_config,
                    prog_tokenizer=program_tokenizer,
                    grid_tokenizer=grid_tokenizer)
model.to(device)
model.train()

if USE_COMPILE:
    if DYNAMIC_BATCHING:
        model = torch.compile(model, dynamic=True)
    else:
        model = torch.compile(model)

optimizer = model.get_optimizer(model_weight_decay=MODEL_WD,
                                model_lr=MODEL_LR,
                                prog_lr_scale=PROGRAM_SCALE,
                                device_type=device.type)


total_time = 0 
total_tokens = 0

for step, batch in enumerate(train_dl):
    t0 = time.time()
    (p, i), o = batch

    bs, sl = o.shape
    p = p.to(device)
    i = i.to(device)
    o = o.to(device)

    optimizer.zero_grad()
    with torch.autocast(device_type= 'cpu' if device.type == 'mps' else device.type, dtype=torch.bfloat16):
        logits = model(p, i)
        loss = model.loss_fn(logits, o)

    loss.backward()
    optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds

    tokens_processed = o.numel()
    tokens_per_sec = tokens_processed / dt

    if step != 0:
        total_time += dt
        total_tokens += tokens_processed

    if device.type == "mps":
        torch.mps.empty_cache()

    print(f"step {step:5d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | BS: {bs:3d} | SL: {sl:5d} | TOKENS: {tokens_processed}")

    if step == 50:
        break


print("Average Tokens/Sec", total_tokens / total_time)
# %%

torch.cuda.empty_cache()
# %%
