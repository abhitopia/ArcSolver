#%%
import sys


src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory
#%%

import random
from src.tokenizer import ArcTokenizer
from src.repl import REPL, REPLConfig
import time
import pandas as pd
import wandb
import numpy as np
import torch
from src.trainer import TrainerBase
from src.arc_hparams import ArcHparams

#%%


experiment = 'development'
run = 'this_run'
seed = 42
device = None
eval_int = 100
grok_alpha = 0.5
grok_lambda = 0.5

hparams = ArcHparams(experiment=experiment,
                    run=run, 
                    seed=seed, 
                    device=device, 
                    eval_interval=eval_int,
                    grok_alpha=grok_alpha,
                    grok_lambda=grok_lambda)

data_config = {
    'include_eval': False,
    'num_train_per_program': 10,
    'max_test_per_program': 3,
}


model_config = {
    "n_dim": 128,
    "n_embd": 16,
    "n_head": 4,
    "n_layer": 3,
    "pnorm": 1.0,
    "dropout": 0.0,
    "n_state_layer": 3,
    "num_iters": 8,
}

optimizer_config = {
    # Batch Size
    "train_batch_token_count": 10000,
    "eval_batch_token_count": 10000,
    "batch_min_util": 0.7,

    # Regularization / Weight Decay
    "dropout": 0.0,
    "wd_model": 0.1,
    "wd_prog": 0.1,
    "l1_prog": 0.0,

    # Loss
    "edr": 2,
    "mctp": 0.4,

    # Learning Rate
    "lr_model": 0.001,
    "lr_prog": 0.001,
    "lr_schedule": "noam",
    "lr_warmup_steps": 2_000,
    "lr_decay_steps": 30_000,  
}

hparams.add_params(prefix="data", **data_config)
hparams.add_params(prefix="optim", **optimizer_config)
hparams.add_params(prefix="model", **model_config)


# train_dl, eval_dl = hparams.init_dataloaders()
# model = hparams.init_model()
# loss = hparams.init_loss_fn()
# optimiser = hparams.init_optimizer(model)
# scheduler = hparams.init_scheduler(optimiser)
#%%

class ArcTrainer(TrainerBase):

    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize() # wait for the GPU to finish work
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            torch.mps.synchronize() # wait for the MPS to finish work
            torch.mps.empty_cache() # clear the MPS cache

    def at_training_start(self):
        self.tokenizer = self.hparams.state['tokenizer']
        self.checkpoint_metric = 'SampleAcc(%)'
        self.console_metrics = self.console_metrics.union({'SampleAcc(%)', 'TokenAcc(%)', '#Loops', 'ΔT(ms)', '#TokensPerSec'})
        self.checkpoint_metric_increases = True
        
        if not self.disable_checkpointing_and_logging:
            # Log params every 10 epochs
            wandb.watch(self.model, log='all', log_freq=max(len(self.train_dl)*10, 500)) 

    def at_epoch_start(self):
        self.train_stats.epoch_reset()

    def at_epoch_end(self):
        self.clear_gpu_cache()

        if self.disable_checkpointing_and_logging:
            return
        
        # Log Sparsity of the Program Embeddings
        threshold = 1e-5 
        sparsity = (self.model.pte[0].weight.abs() < threshold).float().mean().item()
        wandb.log({'Sparsity/Program': sparsity}, step =self.step, commit=False)


    def at_eval_start(self):
        self.eval_stats.epoch_reset()
    
    def at_eval_end(self):
        self.clear_gpu_cache()
        if self.disable_checkpointing_and_logging:
            return

    def pre_train_step(self, batch):
        self.__train_batch_time_start = time.time()

    def post_optimizer_step(self):
        if self.model.config.pnorm is not None:
            target_norm = self.model.config.pnorm
            with torch.no_grad():
                prog_embedding = self.model.pte
                if prog_embedding.weight.grad is not None and prog_embedding.weight.grad.is_sparse:
                    grad = prog_embedding.weight.grad.coalesce()  # Convert sparse gradient to coalesced form
                    indices = grad.indices().squeeze()  # Get the indices of the embeddings that were updated
                    weights = prog_embedding.weight[indices]  # Get the updated embedding vectors

                    # Normalize the L2 norm of each updated embedding vector
                    norm = weights.norm(p=2, dim=1, keepdim=True)
                    scaling_factor = target_norm / norm
                    prog_embedding.weight.data[indices] = weights * scaling_factor  # Correctly rescale the weights

    def pre_eval_step(self, batch):
        self.__eval_batch_time_start = time.time()
    
    def _accuracy(self, logits, y):
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        output_mask = y != self.model.PAD_IDX
        mask_correct_tokens = correct_token_predictions & output_mask
        mask_correct_samples = output_mask.sum(axis=1) == mask_correct_tokens.sum(axis=1)
        return mask_correct_tokens, mask_correct_samples

    @torch.no_grad()
    def _add_step_metrics(self, loss, x, y, iter_logits, is_train):
        metrics_obj = self.train_metrics if is_train else self.eval_metrics
        metrics_obj.add_metric('Loss', loss.item())

        for i, logits in enumerate(iter_logits):
            correct_tokens_mask, correct_samples_mask = self._accuracy(logits, y.target_grid)
            metrics_obj.add_metric(
                    f'{i+1}_TokenAcc',
                    correct_tokens_mask.sum().item(),
                    y.grid.numel())
            
            metrics_obj.add_metric(
                    f'{i+1}_SampleAcc',
                    correct_samples_mask.sum().item(), 
                    y.grid.size(0))
            
            if i == len(iter_logits) - 1:
                metrics_obj.add_metric('TokenAcc(%)', correct_tokens_mask.sum().item() * 100, y.grid.numel())
                metrics_obj.add_metric('SampleAcc(%)', correct_samples_mask.sum() * 100, y.grid.size(0) )

        metrics_obj.add_metric('BatchSize(#Tokens)', y.grid.numel())
        metrics_obj.add_metric('#Samples', y.grid.size(0))
        metrics_obj.add_metric('SeqLen', y.grid.size(1))

    def train_step(self, batch):
        x, y = batch
        iter_logits, _ = self.model(x, y)
        loss = self.loss_fn(iter_logits, y.target_grid)
        self._add_step_metrics(loss, x, y, iter_logits, is_train=True)
        return loss
    
    def eval_step(self, batch):
        x, y = batch
        iter_logits, _ = self.model(x, y)
        loss = self.loss_fn(iter_logits, y.target_grid)
        self._add_step_metrics(loss, x, y, iter_logits, is_train=False)
        return loss
    
    def post_train_step(self, batch):
        x, y = batch
        num_tokens = x.grid.size(8) * (x.grid.size(1) + 3) + y.grid.numel()
        train_batch_time = (time.time() - self.__train_batch_time_start)*1000
        self.train_metrics.add_metric('ΔT(ms)', train_batch_time)
        self.train_metrics.add_metric('#TokensPerSec', num_tokens, (train_batch_time / 1000))

        if self.step % self.hparams.optim.clear_cache_interval == 0:
            self.clear_gpu_cache()
        
    def post_eval_step(self, batch):        
        x, y = batch
        num_tokens = x.grid.size(8) * (x.grid.size(1) + 3) + y.grid.numel()
        eval_batch_time = (time.time() - self.__eval_batch_time_start)*1000
        self.eval_metrics.add_metric('ΔT(ms)', eval_batch_time)
        self.eval_metrics.add_metric('#TokensPerSec', num_tokens, (eval_batch_time / 1000))

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['model_config'] = self.model.config.to_dict()
        state_dict['tokenizer'] = self.hparams.state['tokenizer'].to_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, load_model=True, load_optim=True, strict=True):
        tokenizer = ArcTokenizer.from_dict(state_dict['tokenizer'])
        model_config = REPLConfig.from_dict(state_dict['model_config'])

        
        if strict:
            assert model_config == self.model.config, "Model Configs do not match!"
            assert tokenizer == self.hparams.state['tokenizer'], "Tokenizers do not match!"

        super().load_state_dict(state_dict, load_model=True, load_optim=load_optim, strict=strict)

        if load_model:
            self.info("Copying program embeddings from the loaded model.")
            src_sd = state_dict['model_state_dict']
            trg_prog_token2idx = self.hparams.state['tokenizer'].program_tokenizer.token2idx
            src_prog_token2idx = tokenizer.program_tokenizer.token2idx
            self.model.load_prog_embeddings(trg_prog_token2idx, src_sd, src_prog_token2idx)
            if self.hparams.state['tokenizer'] != tokenizer:
                self.warning("Loaded model has different tokenizers than the current model. Loading anyway as the models are compatible.")
                self.warning("If this is not intened, stop and re-evaluate the situation.")
        self._eval_at_start = True

trainer = ArcTrainer(
    hparams=hparams,
    parent_dir='/tmp/arc_solver',
    prevent_overwrite=True,
    num_checkpoints_to_keep=4,
    disable_checkpointing_and_logging=True
)

trainer_sd = trainer.state_dict()
trainer.load_state_dict(trainer_sd, load_model=True, load_optim=True, strict=True)

# trainer.at_training_start()
# train_dl, eval_dl = trainer.train_dl, trainer.eval_dl


# batch = next(iter(train_dl))
# trainer.train_step(batch)
#%%
# print(trainer.state_dict())


# %%
