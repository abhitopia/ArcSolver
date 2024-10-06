from bisect import bisect_right
from collections import defaultdict
import time
import numpy as np
import torch
import wandb

from .repl import REPLConfig
from .tokenizer import ArcTokenizer
from .trainer import TrainerBase


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
        self.n_iter = self.hparams.optim['n_iter']

        ## Set up complexity levels
        self.num_levels = 5
        complexities = [m['complexity'] for x, y in self.train_dl for m in x.meta]
        quantiles = np.linspace(0, 1, (self.num_levels + 1))
        self.level_edges = np.quantile(complexities, quantiles)
        self.level_edges[0] = 0
        self.level_edges[-1] = 1.0

        
        if not self.disable_checkpointing_and_logging:
            # Log params every 10 epochs
            wandb.watch(self.model, log='all', log_freq=max(len(self.train_dl)*10, 500)) 

    def at_epoch_end(self):
        self.clear_gpu_cache()

        if self.disable_checkpointing_and_logging:
            return
        
        # Log Sparsity of the Program Embeddings
        threshold = 1e-5 
        sparsity = (self.model.pte[0].weight.abs() < threshold).float().mean().item()
        wandb.log({'Sparsity/Program': sparsity}, step =self.step, commit=False)

    def at_eval_end(self):
        self.clear_gpu_cache()

    def pre_train_step(self, batch):
        self.__train_batch_time_start = time.time()


    def post_optimizer_step(self):
        if self.model.config.pnorm is not None:
            target_norm = self.model.config.pnorm
            with torch.no_grad():
                prog_embedding = self.model.pte[0]
                if prog_embedding.weight.grad is not None:
                    grad = prog_embedding.weight.grad  # Dense gradient
                    # Identify the embeddings that were updated (non-zero gradients)
                    mask = grad.abs().sum(dim=1) != 0
                    # indices = mask.nonzero(as_tuple=False).squeeze()
                    indices = mask.nonzero(as_tuple=True)[0]  # Ensure indices is 1D


                    if indices.numel() > 0:
                        weights = prog_embedding.weight.data[indices]  # Get the updated embedding vectors
                        # Normalize the L2 norm of each updated embedding vector
                        norm = weights.norm(p=2, dim=1, keepdim=True)
                        scaling_factor = target_norm / norm
                        prog_embedding.weight.data[indices] = weights * scaling_factor  # Rescale the weights

    def pre_eval_step(self, batch):
        self.__eval_batch_time_start = time.time()
    
    def _accuracy(self, logits, y):
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        output_mask = y != self.model.PAD_IDX
        mask_correct_tokens = correct_token_predictions & output_mask
        mask_correct_samples = output_mask.sum(axis=1) == mask_correct_tokens.sum(axis=1)
        total_tokens = output_mask.sum().item()
        return mask_correct_tokens, mask_correct_samples, total_tokens

    @torch.no_grad()
    def _add_step_metrics(self, loss, x, y, iter_logits, is_train):
        metrics_obj = self.train_metrics if is_train else self.eval_metrics
        metrics_obj.add_metric('Loss', loss.item())

        # Extract the dataset names and  complexities for each sample in the batch
        datasets = [meta['dataset'] for meta in x.meta]
        complexities = [meta['complexity'] for meta in x.meta]
        levels = [min(self.num_levels-1, max(0, bisect_right(self.level_edges, c) - 1)) for c in complexities]

        # Create mappings for dataset indices, level indices, and level-dataset indices
        dataset_indices = defaultdict(list)
        level_indices = defaultdict(list)
        level_dataset_indices = defaultdict(list)
        for idx, (dataset, level) in enumerate(zip(datasets, levels)):
            dataset_indices[dataset].append(idx)
            level_name = f"L{level+1}"
            level_indices[level_name].append(idx)
            level_dataset_indices[(level_name, dataset)].append(idx)


        for i, logits in enumerate(iter_logits):
            correct_tokens_mask, correct_samples_mask, total_tokens = self._accuracy(logits, y.target_grid)
            num_tokens_correct = correct_tokens_mask.sum().item()
            num_samples_correct = correct_samples_mask.sum().item()
            total_samples_batch = y.grid.size(0)
            metrics_obj.add_metric(
                    f'TokenAcc/I{i+1}',
                    num_tokens_correct,
                    total_tokens)
            
            metrics_obj.add_metric(
                    f'SampleAcc/I{i+1}',
                    num_samples_correct, 
                    total_samples_batch)
                

            # Only for last iteration!
            if i == len(iter_logits) - 1:
                metrics_obj.add_metric('TokenAcc(%)', num_tokens_correct * 100, total_tokens)
                metrics_obj.add_metric('SampleAcc(%)', num_samples_correct * 100, total_samples_batch)

                # Compute Sample Accuracy per level
                for level, indices in level_indices.items():
                    correct_samples_in_level = correct_samples_mask[indices]
                    num_correct = correct_samples_in_level.sum().item()
                    total_samples = len(indices)
                    metrics_obj.add_metric(
                        f'LevelAcc(%)/{level}',
                        num_correct * 100,
                        total_samples)

                for dataset, indices in dataset_indices.items():
                    if not indices:
                        continue  # Skip if there are no samples for this dataset
                    correct_samples_in_dataset = correct_samples_mask[indices]
                    num_correct = correct_samples_in_dataset.sum().item()
                    total_samples = len(indices)
                    metrics_obj.add_metric(
                        f'{dataset}/Accuracy(%)',
                        num_correct * 100,
                        total_samples 
                    )

                # Compute Sample Accuracy per level per dataset
                for (level, dataset), indices in level_dataset_indices.items():
                    correct_samples_in_group = correct_samples_mask[indices]
                    num_correct = correct_samples_in_group.sum().item()
                    total_samples = len(indices)
                    metrics_obj.add_metric(
                        f'{dataset}/{level}',
                        num_correct,
                        total_samples)
            

        metrics_obj.add_metric('BatchSize(#Tokens)', y.grid.numel())
        metrics_obj.add_metric('#Samples', y.grid.size(0))
        metrics_obj.add_metric('SeqLen', y.grid.size(1))

    def train_step(self, batch):
        x, y = batch
        iter_logits, _ = self.model(x, y,  self.n_iter)
        loss = self.loss_fn(iter_logits, y.target_grid)
        self._add_step_metrics(loss, x, y, iter_logits, is_train=True)
        return loss
    
    def eval_step(self, batch):
        x, y = batch
        iter_logits, _ = self.model(x, y, self.n_iter)
        loss = self.loss_fn(iter_logits, y.target_grid)
        self._add_step_metrics(loss, x, y, iter_logits, is_train=False)
        return loss
    
    def post_train_step(self, batch):
        x, y = batch
        num_tokens = x.grid.size(0) * (x.grid.size(1) + 3) + y.grid.numel()
        train_batch_time = (time.time() - self.__train_batch_time_start)*1000
        self.train_metrics.add_metric('ΔT(ms)', train_batch_time)
        self.train_metrics.add_metric('#TokensPerSec', num_tokens, (train_batch_time / 1000))

        if self.step % self.hparams.optim.clear_cache_interval == 0:
            self.clear_gpu_cache()
        
    def post_eval_step(self, batch):        
        x, y = batch
        num_tokens = x.grid.size(0) * (x.grid.size(1) + 3) + y.grid.numel()
        eval_batch_time = (time.time() - self.__eval_batch_time_start)*1000
        self.eval_metrics.add_metric('ΔT(ms)', eval_batch_time)
        self.eval_metrics.add_metric('#TokensPerSec', num_tokens, (eval_batch_time / 1000))

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['model_config'] = self.model.config.to_dict()
        state_dict['tokenizer'] = self.hparams.state['tokenizer'].to_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, load_model=True, load_step=True, load_optim=True, strict=True):
        tokenizer = ArcTokenizer.from_dict(state_dict['tokenizer'])
        model_config = REPLConfig.from_dict(state_dict['model_config'])
        
        if strict:
            assert model_config == self.model.config, "Model Configs do not match!"
            assert tokenizer == self.hparams.state['tokenizer'], "Tokenizers do not match!"

        super().load_state_dict(state_dict, load_model=True, load_step=load_step, load_optim=load_optim, strict=strict)

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