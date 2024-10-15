from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from .deploy_utils import AdamWModule, format_float, generate_lr_schedule, shuffled_indices, split_task, Task, TaskSolution, MODEL_INPUT, MODEL_OUTPUT, loss_fn, deserialize_array
from .repl import REPL, REPLConfig


class SolverParams(NamedTuple):
    thinking: int = 500
    bs: int = 25
    patience: int = 30
    lr: float = 0.005
    wd: float = 0.05
    wu: int = 10
    lrs: float = 0.1
    seed: int = 60065
    mode: str = '60065'
    confidence: float = 0.0001
    metric: str = 'L'


class Solver(nn.Module):
    def __init__(self, model: torch.ScriptModule) -> None:
        super().__init__()
        self.model = model
        self.adam = AdamWModule(self.model.get_pte_weight(),
                                betas=(0.9, 0.95),
                                eps=1e-8) 
        
        self.inner_step = 0
        self.step = 0
        self.verbose = False

        pte = self.model.get_pte_weight()
        self._init = torch.zeros_like(pte, requires_grad=False)
        self._init.data.copy_(pte.data)
        self.solution = torch.zeros_like(pte, requires_grad=False)
        self.min_loss = float('inf')
        self.bad_steps = 0
        self.patience = -1
        self.bs = 5
        self.print_prefix = ""

    def print(self, msg: str):
        if self.verbose:
            print(f"{self.print_prefix}:\t{msg}")

    def reset(self):
        self.adam.reset()
        self.step = 0
        self.inner_step = 0
        self.model.get_pte_weight().data.copy_(self._init.data)
        self.solution.zero_()
        self.min_loss = float('inf')
        self.bad_steps = 0

    def model_updated(self) -> bool:
        return self.inner_step % self.bs  == self.bs - 1

    def update_solution(self, me: Dict[str, float], metric: str) -> None:
        if not self.model_updated():
            return
        
        loss = me[metric]
        if loss < self.min_loss:
            self.min_loss = loss
            self.solution.data.copy_(self.model.get_pte_weight().data)
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        
    def train_step(self, x: MODEL_INPUT, y: Optional[MODEL_OUTPUT], lr: float, wd: float):
        if self.inner_step % self.bs == 0:
            self.adam.zero_grad()
        logits, _ = self.model(x, y)
        assert y is not None
        loss = loss_fn(logits, y)
        loss = loss / self.bs # Gradient accumulation
        loss.backward()
        if self.model_updated():
            self.adam.step(lr=lr, wd=wd)
            self.step += 1
        self.inner_step += 1

    def metric(self, logits: Tensor, y: MODEL_OUTPUT) -> Tuple[int, int, int]:
        y: Optional[Tensor] = y.target_grid if y.target_grid is not None else None
        assert y is not None
        _, predicted_tokens = torch.max(logits, dim=2)
        correct_token_predictions = (predicted_tokens == y)
        output_mask = y != 0
        mask_correct_tokens = correct_token_predictions & output_mask
        mask_correct_samples = output_mask.sum(dim=1) == mask_correct_tokens.sum(dim=1)
        num_correct_tokens = mask_correct_tokens.sum().item()
        num_correct_samples = mask_correct_samples.sum().item()
        num_tokens  = output_mask.sum().item()
        return num_correct_tokens, num_correct_samples, num_tokens

    def evaluate(self, examples: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]]) -> Dict[str, float]:

        # Only evaluate after at the end of gradient update
        with torch.no_grad():
            if not self.model_updated():
                return {}
            
            if examples[0][1] is None:
                return {}

            total_loss = 0.0
            total_tokens = 0.0
            total_correct_samples = 0.0
            total_correct_tokens = 0.0
            max_loss = float('-inf')
            for x, y in examples:
                logits, _ = self.model(x, y)
                assert y is not None
                loss = loss_fn(logits, y)

                max_loss = max(max_loss, loss.item())
                num_correct_tokens, num_correct_samples, num_tokens = self.metric(logits, y)

                total_correct_samples += num_correct_samples
                total_correct_tokens += num_correct_tokens
                loss_sum = loss.item() * num_tokens
                total_loss += loss_sum
                total_tokens += num_tokens
  
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            sample_accuracy = total_correct_samples / len(examples)
            token_accuracy = total_correct_tokens / total_tokens

            metrics: Dict[str, float] = {
                'L': avg_loss,
                'ML': max_loss,
                'TA': token_accuracy,
                'SA': sample_accuracy,
            }   
            return metrics
        
    def log_stats(self, me: Dict[str, float], mt: Dict[str, float]) -> None:
        if not self.model_updated():
            return
        
        if not self.verbose:
            return
        
        step: str = str(self.step)
        step = ''.join([' ' for _ in range(max(0, 3 - len(step)))]) + step
        
        metrics: Dict[str, str] = {
            'T': step,
            'EL': format_float(me['L'], 4),
            'EML': format_float(me['ML'], 4),
            'ETA': format_float(me['TA'], 3),
            'ESA': format_float(me['SA'], 2),
            'MML': format_float(self.min_loss, 4),
            'RS': str(self.patience - self.bad_steps),
        }

        if len(mt) > 0:
            metrics['TL'] = format_float(mt['L'], 4)
            metrics['TML'] = format_float(mt['ML'], 4)
            metrics['TTE'] = format_float(mt['TA'], 3)
            metrics['TSE'] = format_float(mt['SA'], 2)

        msg = ""
        for k, v in metrics.items():
            msg += f"{k}: {v} "

        self.print(msg)
        
    def forward(self, 
            task: Task, 
            params: SolverParams,
        )-> TaskSolution:
        
        self.verbose = True if params.mode == 'vbs' else False
        torch.manual_seed(params.seed)
        self.bs = params.bs
        self.print(f"Params: {params}")
        self.patience = params.patience
        self.reset()
        device = str(self.model.get_pte_weight().device)

        train_examples, eval_examples, test_examples = split_task(task, device=device)
        self.print_prefix = f"Task {task.task_id}"
        self.print(f"Device: {device}")
        self.print(f"# TRAIN: {len(train_examples)}")
        self.print(f"# EVAL: {len(eval_examples)}")
        self.print(f"# TEST: {len(test_examples)}")

        lr_schedule = generate_lr_schedule(params.lr, params.wu, params.thinking, params.lrs)

        self.print(f"Thinking...")
        while True:
            for idx in shuffled_indices(len(train_examples)):
                batch = train_examples[idx]
                x: MODEL_INPUT = batch[0]
                y: Optional[MODEL_OUTPUT] = batch[1]

                lr_step = lr_schedule[self.step]
                self.train_step(x, y, lr=lr_step, wd=params.wd)
                me = self.evaluate(eval_examples)
                self.update_solution(me, params.metric)
                mt = self.evaluate(test_examples)
                self.log_stats(me, mt)

                if self.step == params.thinking or self.bad_steps >= params.patience:
                    self.print(f"Bad Steps: {self.bad_steps}")
                    break
            
            if self.step == params.thinking or self.bad_steps >= params.patience:
                break

        preds, scores = self.predict(test_examples, params.confidence)
        solution = TaskSolution(task.task_id, preds, scores)

        return solution


    def predict(self, test_examples: List[Tuple[MODEL_INPUT, Optional[MODEL_OUTPUT]]], confidence: float)-> Tuple[List[List[Tensor]], List[List[float]]]:
        self.print("Generating predictions ...")
        # Load the best solution
        self.model.get_pte_weight().data.copy_(self.solution.data)

        preds: List[List[Tensor]] = []
        scores: List[List[float]] = []
        for eid, example in enumerate(test_examples):
            x, y = example
            grid: List[int] = x.grid[0].tolist()
            indices: List[List[int]] = x.grid_indices[0].tolist()
            # gp, gs = self.model.greedy_search(grid, indices)
            # bps, bss = [gp], [gs]
            bps, bss = self.model.beam_search(grid, 
                                              indices,
                                              prob_thresh=confidence)
            
            self.print(f"Processed Test input: {eid + 1}")
            pred_tensors: List[Tensor] = []
            score_tensors: List[float] = []
            for bp, bs in zip(bps, bss):
                pred_tensors.append(deserialize_array(bp, device='cpu'))
                score_tensors.append(bs)

            preds.append(pred_tensors)
            scores.append(score_tensors)
            solved = False
            if y is not None:
                target_grid = y.grid
                target: List[int] = target_grid[0].tolist()
                for idx, bp in enumerate(bps):
                    if bp == target:
                        self.print(f"Congrats! Solved test input: {eid + 1} at prediction: {idx + 1}")
                        solved = True
                        break
                if not solved:
                    self.print(f"Unable to solve test input: {eid + 1}")

        return preds, scores


@staticmethod
def load_inference_model(ckt_path, jit: bool = True):
    data = torch.load(ckt_path, map_location='cpu', weights_only=False)
    programs = data['model_state_dict']['pte.0.weight']
    model_config = REPLConfig.from_dict(data['model_config'])
    model_config.prog_vocab_size = 1
    model_config.dropout = 0.0 # this is important for inference as I cannot change dropout in the scripted Solver
    model = REPL(model_config)
    model.train() # This is because RNN only does backward in training mode
    model.state_agg.rnn.flatten_parameters()
    model_state_dict = {k: v for k, v in data['model_state_dict'].items() if 'pte.0.weight' not in k}
    model_state_dict['pte.0.weight'] = model.pte[0].weight
    model.load_state_dict(model_state_dict, strict=True)
    for name, param in model.named_parameters():
        if 'pte.0.weight' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.pte[0].weight.data.copy_(programs.data.mean(dim=0))
    if jit:
        model = torch.jit.script(model)
    return model 


def create_solver(
        ckpt_path: str,
        jit=True,
        save_path=None
    ) -> Solver:

    model = load_inference_model(ckpt_path, jit=jit)
    solver = Solver(model)
    if jit:
        solver = torch.jit.script(solver)
        if save_path is not None:
            print(f"Saving the model to {save_path}")
            torch.jit.save(solver, save_path)
    return solver


        
