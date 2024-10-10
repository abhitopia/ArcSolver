#%%
from dataclasses import dataclass
import functools
from itertools import product
import random
from typing import Any, List

import torch
import torch.nn.functional as F
from src import tokenizer
from src.arc_hparams import noam_schedule
from src.dataset import ArcExamplesDataset
from src.lazy_adamw import LazyAdamW
from src.lrscheduler import LambdaLRWithReduceOnPlateau
from src.repl import REPL, REPLConfig
from src.tokenizer import ArcTokenizer
from src.task import ARC_SYNTH, ArcTask, ArrayTransform, ColorPermutation, Example, ARC_EVAL
from src.utils import map_to_tensors
#%%
# loader = ARC_SYNTH
loader = ARC_EVAL
tasks = loader.tasks
#%%

@dataclass
class ArcSolverConfig:
    ckt_path: str

    # Data
    n_train_ex: int = 20
    tbs: int = 5
    ebs: int = 5
    # permute: bool = True

    # Train
    n_grad_accum: int = 1
    
    # Optimizer
    lr: float = 0.001
    wd: float = 0.05
    dropout: float = 0.01

    # Initialization
    init_method: str = 'zero'    # Can be 'zero' and 'avg'
    jit: bool = False

    # Scheduler
    warmup_steps: int = 500
    decay_steps: int = 5000
    min_lr_scale: float = 0.01

    plt_metric: str = 'accuracy'

    plt_patience: int = 2
    plt_mode: str = 'max'
    plt_warmup: int = 10
    plt_factor: float = 0.8

    # def __post_init__(self):
    #     assert self.bs <= self.n_train_ex, 'Batch size must be less than or equal to the number of training examples'


class ARCSolver:
    def __init__(self, config: ArcSolverConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(config)
        self.model.to(self.device)
        self.tokenizer = self.load_tokenizer(config)
        self.pad_idx = self.tokenizer.grid_tokenizer.PAD_IDX
        self.train_perms, self.eval_perms = self.get_augmentation_split()
        self.reset()

    def reset(self):
        if self.config.init_method == 'zero':
            self.model.pte[0].weight.data.zero_()

        self.metrics = {}
        self.optim = self.optimizer(config, self.model)
        self.scheduler = self.load_scheduler(config, self.optim)

    @staticmethod
    def get_augmentation_split():
        tforms = list(ArrayTransform)
        cps = list(ColorPermutation)
        augs = list(product(tforms, cps))

        test_cps = set()
        test_tforms = set()

        eval_tforms = []
        train_tforms = []
        for tform, cp in augs[1:]:
            if cp not in test_cps and tform not in test_tforms:
                test_cps.add(cp)
                test_tforms.add(tform)
                eval_tforms.append((tform, cp))
            else:
                train_tforms.append((tform, cp))

            if len(test_cps) == len(cps) and len(test_tforms) == len(tforms):
                break
            
        return train_tforms, eval_tforms


    @staticmethod
    def load_model(config):
        ckt_path = config.ckt_path
        data = torch.load(ckt_path, map_location='cpu', weights_only=False)
        model_config = REPLConfig.from_dict(data['model_config'])
        model_config.prog_vocab_size = 1
        model_config.dropout = config.dropout
        model = REPL(model_config)
        model_state_dict = {k: v for k, v in data['model_state_dict'].items() if 'pte.0.weight' not in k}
        model_state_dict['pte.0.weight'] = model.pte[0].weight
        model.load_state_dict(model_state_dict, strict=True)
        for name, param in model.named_parameters():
            if 'pte.0.weight' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        return model

    @staticmethod
    def load_tokenizer(config):
        ckt_path = config.ckt_path
        data = torch.load(ckt_path, map_location='cpu', weights_only=False)
        tokenizer = ArcTokenizer.from_dict(data['tokenizer'])
        return tokenizer

    @staticmethod
    def optimizer(config, model):
        program_params = [p for p in model.parameters() if p.requires_grad]
        optim_groups = [
            {
                'params': program_params,
                'lr': config.lr,
                'weight_decay': config.wd,
                'l1_coeff': 0.0
            }
        ]

        use_fused = torch.cuda.is_available()
        optimizer = LazyAdamW(optim_groups,
                            lr=config.lr,
                            weight_decay=config.wd, 
                            betas=(0.9, 0.95), eps=1e-8,
                            fused=use_fused)
        return optimizer

    @staticmethod
    def load_scheduler(config, optimizer):

        schedule = lambda step: noam_schedule(
                                        step, 
                                        warmup_steps=config.warmup_steps, 
                                        decay_steps=config.decay_steps,
                                        min_lr_scale=config.min_lr_scale
                                    )

        scheduler = LambdaLRWithReduceOnPlateau(
            optimizer,
            lr_lambda=schedule,
            mode=config.plt_mode,
            factor=config.plt_factor,
            patience=config.plt_patience,
            warmup_epochs=config.plt_warmup,
            verbose=False
        )

        return scheduler


    def _loss_fn(self, logits, y):
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none').view_as(y)  
        mask = y != self.pad_idx
        loss = (loss * mask).sum() / mask.sum()

        loss_sum = (loss * mask).sum()
        num_tokens = mask.sum()
        return loss_sum, num_tokens
    
    def _accuracy_fn(self, logits, y):
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        output_mask = y != self.pad_idx
        mask_correct_tokens = correct_token_predictions & output_mask
        mask_correct_samples = output_mask.sum(axis=1) == mask_correct_tokens.sum(axis=1)
        total_tokens = output_mask.sum().item()
        num_correct_tokens = mask_correct_tokens.sum().item()
        num_correct_samples = mask_correct_samples.sum().item()
        total_samples = y.size(0)
        return num_correct_tokens, total_tokens, num_correct_samples, total_samples
    

    def _train_step(self, batch, accum_step):
        self.model.train()
        x, y = map_to_tensors(batch, lambda x: x.to(self.device, non_blocking=True))

        if accum_step == 0:
            self.optim.zero_grad()

        logits, _ = self.model(x, y)
        loss_sum, num_tokens = self._loss_fn(logits[-1], y.target_grid)
        loss = loss_sum / num_tokens
        loss_accum = loss / self.config.n_grad_accum
        loss_accum.backward()

        if accum_step == self.config.n_grad_accum - 1:
            self.optim.step()
            self.scheduler.step()

        self.metrics['STL'] =  (loss.item(), 3)
        return loss.item()

    def evaluate(self, dl, prefix):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        total_correct_tokens = 0
        total_samples = 0
        total_correct_samples = 0
        num_batches = len(dl)
        for batch in dl:
            x, y = map_to_tensors(batch, lambda x: x.to(self.device, non_blocking=True))
            with torch.no_grad():
                iter_logits, _ = self.model(x, y)
                logits = iter_logits[-1]
                loss, _ = self._loss_fn(logits, y.target_grid)

            nct, tt, ncs, ts = self._accuracy_fn(logits, y.target_grid)

            # Update metrics
            total_tokens += tt
            total_correct_tokens += nct
            total_samples += ts
            total_correct_samples += ncs
            total_loss += loss.item()



        avg_loss = total_loss / total_tokens if num_batches > 0 else 0
        token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
        sample_acc = total_correct_samples / total_samples if total_samples > 0 else 0

        self.metrics[f'{prefix}L'] = (avg_loss, 4)
        self.metrics[f'{prefix}TA'] = (token_acc, 3)
        self.metrics[f'{prefix}SA'] = (sample_acc, 2)

        return avg_loss
            

    def get_collate_fn(self, permute=False):
        collate_fn = functools.partial(ArcExamplesDataset.collate_fn,
                            pad_idx=self.pad_idx, 
                            tokenizer=self.tokenizer, 
                            prog_idx = 0, 
                            permute=permute)
        return collate_fn
        


    def get_dataloader(self, examples: List[Example], bs, permute):
        train_ds = ArcExamplesDataset(examples, tokenizer)
        train_dl = torch.utils.data.DataLoader(train_ds,
                                            batch_size=bs, 
                                            shuffle=True, 
                                            drop_last=False,
                                            collate_fn=self.get_collate_fn(permute))
        return train_dl

    def get_train_eval_split(self, examples: List[Example]):
        train_examples = []

        train_examples = [ex.clone().permute(cp, tf) for ex in examples for tf, cp in self.train_perms]
        eval_examples = [ex.clone().permute(cp, tf) for ex in examples for tf, cp in self.eval_perms]
        train_examples += examples

        return train_examples, eval_examples
    

    def log_metrics(self):
        metric_str = ''
        for key, (value, p) in self.metrics.items():
            if isinstance(value, float):
                metric_str += f'{key}: {value:.{p}f} '
            elif isinstance(value, int):
                metric_str += f'{key}: {value:{p}d} '
            else:
                metric_str += f'{key}: {value} '
        print(metric_str)

    def __call__(self, task: ArcTask) -> Any:
        self.reset()
        train_examples, eval_examples = self.get_train_eval_split(task.train)
        test_examples = task.test
        train_dl = self.get_dataloader(train_examples, self.config.tbs, permute=True)
        eval_dl = self.get_dataloader(task.train, self.config.ebs, permute=False)
        test_dl = self.get_dataloader(test_examples, self.config.ebs, permute=False)

        total_step = self.config.warmup_steps + self.config.decay_steps
        step = 0

        print("Task", task.id)
        print("Program ID", task.prog_id)
        print("Num Train Examples", len(task.train))
        print("Num Test Examples", len(task.test))
        print("Num Augmented Examples", len(train_examples))
        self.evaluate(test_dl, 'TE')

        self.log_metrics()
        while True:
            for idx, batch in enumerate(train_dl):
                self.metrics = {'T': (step, 3)}
                accum_step = step % self.config.n_grad_accum
                loss = self._train_step(batch, accum_step)
                if accum_step == self.config.n_grad_accum - 1:
                    self.evaluate(eval_dl, 'TRA')
                    self.evaluate(test_dl, 'TE')
                    self.scheduler.step_metric(self.metrics['TRATA'][0])
                    self.metrics['lr'] = (self.scheduler.get_last_lr()[0], 4)
                    self.log_metrics()

                step += 1
                if step >= total_step:
                    break

            if step >= total_step:
                break


ckt_path = '/teamspace/studios/work-horse/ArcSolver/runs/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
# ckt_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver/models/v9/D512E128H16B5I3.v1/ckt_162000_39.205.pth'
config = ArcSolverConfig(
    lr=0.01,
    wd=0.05,
    dropout=0.01,
    min_lr_scale=0.01,
    tbs = 10,
    ebs = 10,
    n_grad_accum=5,
    warmup_steps=50,
    decay_steps=1000,
    ckt_path=ckt_path,
    plt_factor=0.5,
    plt_patience=10,
    plt_warmup=0
    )

solve = ARCSolver(config=config)
print("Num Tasks", len(tasks))
# task_id = 1009 #3500
task_id = 111
task = tasks[task_id]
print(task)
solve(task=task)

# %%
