#%%
from dataclasses import dataclass
import functools
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
from src.task import ARC_SYNTH, ArcTask, Example, ARC_EVAL
from src.utils import map_to_tensors
#%%
loader = ARC_SYNTH
tasks = loader.tasks
#%%

@dataclass
class ArcSolverConfig:
    ckt_path: str

    # Data
    n_train_ex: int = 20
    bs = 5
    permute: bool = True

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



    def __post_init__(self):
        assert self.bs <= self.n_train_ex, 'Batch size must be less than or equal to the number of training examples'



class ARCSolver:
    def __init__(self, config: ArcSolverConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(config)
        self.model.to(self.device)
        self.tokenizer = self.load_tokenizer(config)
        self.pad_idx = self.tokenizer.grid_tokenizer.PAD_IDX
        self.reset()

    def reset(self):
        if self.config.init_method == 'zero':
            self.model.pte[0].weight.data.zero_()

        self.metrics = {}
        self.optim = self.optimizer(config, self.model)
        self.scheduler = self.load_scheduler(config, self.optim)

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
        return loss
    
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
        loss = self._loss_fn(logits[-1], y.target_grid)

        loss_accum = loss / self.config.n_grad_accum
        loss_accum.backward()

        if accum_step == self.config.n_grad_accum - 1:
            self.optim.step()
            self.scheduler.step()

        self.metrics['STL'] =  loss.item()
        return loss.item()

    def evaluate(self, examples: List[Example], prefix):
        self.model.eval()
        collate_fn = self.get_collate_fn()
        total_loss = 0
        total_tokens = 0
        total_correct_tokens = 0
        total_samples = 0
        total_correct_samples = 0
        for example in examples:
            collate_ex = collate_fn([example])
            x, y = map_to_tensors(collate_ex, lambda x: x.to(self.device, non_blocking=True))
            with torch.no_grad():
                iter_logits, _ = self.model(x, y)
                logits = iter_logits[-1]
                loss = self._loss_fn(logits, y.target_grid)

            nct, tt, ncs, ts = self._accuracy_fn(logits, y.target_grid)

            # Update metrics
            total_tokens += tt
            total_correct_tokens += nct
            total_samples += ts
            total_correct_samples += ncs
            total_loss += loss.item()

        avg_loss = total_loss / len(examples) if len(examples) > 0 else 0
        token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
        sample_acc = total_correct_samples / total_samples if total_samples > 0 else 0

        self.metrics[f'{prefix}L'] = avg_loss
        self.metrics[f'{prefix}TA'] = token_acc
        self.metrics[f'{prefix}SA'] = sample_acc
        return avg_loss
            


    def get_collate_fn(self):
        collate_fn = functools.partial(ArcExamplesDataset.collate_fn,
                            pad_idx=self.pad_idx, 
                            tokenizer=self.tokenizer, 
                            prog_idx = 0, 
                            permute=self.config.permute)
        return collate_fn
        



    def get_dataloader(self, examples: List[Example]):
        train_ds = ArcExamplesDataset(examples, tokenizer)
        train_dl = torch.utils.data.DataLoader(train_ds,
                                            batch_size=self.config.bs, 
                                            shuffle=True, 
                                            collate_fn=self.get_collate_fn())
        return train_dl

    def get_augment_examples(self, min_num, examples):
        num_to_augment = max(min_num - len(examples), 0)
        num_to_augment = min_num
        augmented_examples = []
        for i in range(num_to_augment):
            example_idx = i % len(examples)
            e = examples[example_idx].clone().permute()
            augmented_examples.append(e)
        return augmented_examples
    

    def log_metrics(self):
        metric_str = ''
        for key, value in self.metrics.items():
            if isinstance(value, float):
                metric_str += f'{key}: {value:.4f}  '
            elif isinstance(value, int):
                metric_str += f'{key}: {value:3d}  '
            else:
                metric_str += f'{key}: {value}  '
        print(metric_str)

    def __call__(self, task: ArcTask) -> Any:
        self.reset()
        train_examples = task.train
        test_examples = task.test
        aux_examples = self.get_augment_examples(self.config.n_train_ex, train_examples)
        train_dl = self.get_dataloader(aux_examples + train_examples)

        total_step = self.config.warmup_steps + self.config.decay_steps
        step = 0

        print("Task", task.id)
        print("Program ID", task.prog_id)
        print("Num Train Examples", len(train_examples))
        print("Num Test Examples", len(test_examples))
        print("Num Augmented Examples", len(aux_examples))

        self.evaluate(test_examples, 'test')
        self.log_metrics()
        while True:
            for idx, batch in enumerate(train_dl):
                self.metrics = {'T': step}
                accum_step = step % self.config.n_grad_accum
                loss = self._train_step(batch, accum_step)
                if accum_step == self.config.n_grad_accum - 1:
                    self.evaluate(train_examples, 'TRA')
                    self.evaluate(test_examples, 'TE')
                    # self.scheduler.step_metric(self.metrics['TRATA'])
                    self.log_metrics()

                step += 1
                if step >= total_step:
                    break

            if step >= total_step:
                break


ckt_path = '/teamspace/studios/work-horse/ArcSolver/runs/v9/D512E128H16B5I3.v1/ckt_281000_52.168.pth'
config = ArcSolverConfig(
    lr=0.005,
    wd=0.5,
    dropout=0.1,
    n_train_ex=20,
    min_lr_scale=0.01,
    bs = 5,
    n_grad_accum=3,
    warmup_steps=30,
    decay_steps=1000,
    ckt_path=ckt_path,
    plt_factor=0.5,
    plt_patience=10,
    plt_warmup=0
    )

solve = ARCSolver(config=config)
print("Num Tasks", len(tasks))
task_id = 3500
task = tasks[task_id]
print(task)
solve(task=task)
