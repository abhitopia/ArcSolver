import math
from typing import Optional, Tuple
import typer

from src.dataset import TrainingData
from src.interpreter import Interpreter, InterpreterConfig
from src.arc_trainer import ArcTrainer
from src.trainer import Hparams
from src.utils import nearest_greater_power_of_2
from rich import print

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")



from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

class SolverHparams(Hparams):

    def init_dataloaders(self)-> Tuple[DataLoader, DataLoader]:

        training_data = TrainingData(
                    augmentation_factor=self.data.data_aug,
                    join_version=not self.data.sep_task_version, 
                    seed=self.seed).load()
        
        # self.state['training_data'] = training_data
        self.state['prog_tokenizer'] = training_data.program_tokenizer
        self.state['grid_tokenizer'] = training_data.grid_tokenizer
        self.state['prog_vocab_size'] = nearest_greater_power_of_2(len(training_data.program_tokenizer))
        self.state['grid_vocab_size'] = nearest_greater_power_of_2(len(training_data.grid_tokenizer))

        # training_data = self.state['training_data']
        config = self.optim
        train_ds = training_data.train_ds.subset(config.max_examples)
        eval_ds = training_data.eval_ds.subset(config.max_examples)
        
        train_dl = train_ds.get_dataloader(batch_size=config.batch_size,
                                           seq_len=self.state['grid_vocab_size'],
                                           batch_by_token_count=True,
                                           pin_memory=True)

        eval_dl = eval_ds.get_dataloader(batch_size=config.batch_size,
                                        seq_len=self.state['grid_vocab_size'],
                                        batch_by_token_count=True,
                                        pin_memory=True)
        
        self.state['num_train_batches'] = len(train_dl)
        self.state['num_eval_batches'] = len(eval_dl)
        return train_dl, eval_dl
    
    def init_model(self)-> nn.Module:
        config = InterpreterConfig(
            prog_vocab_size = self.state['prog_vocab_size'],
            grid_vocab_size = self.state['grid_vocab_size'],
            n_dim = self.model.n_dim, # dimension of the model
            n_head = self.model.n_heads, # number of heads within each self-attention block
            n_mixers = self.model.n_mixers, # number of self-attention layers within each transformer block
            n_blocks = self.model.n_blocks, # number of transformer blocks within each recurrence block
            n_rec_layers = self.model.n_layers, # number of recurrences
            share_mixer = self.model.share_mixer
        )
        model = Interpreter(config,
                            prog_tokenizer=self.state['prog_tokenizer'],
                            grid_tokenizer=self.state['grid_tokenizer'])
        
        self.state['model'] = model
        return model
    
    def init_optimizer(self)-> optim.Optimizer:
        model = self.state['model']
        config = self.optim
        optimizer = model.get_optimizer(
                                    model_lr=config.model_lr,
                                    model_wd=config.model_wd,
                                    prog_lr=config.prog_lr,
                                    prog_wd=config.prog_wd,
                                    device_type=self.device)

        self.state['optimizer'] = optimizer
        return optimizer
    
    def init_scheduler(self)-> optim.lr_scheduler.LambdaLR:
        config = self.optim
        def multiplicative_schedule(step):
            max_lr = 1.0
            min_lr = max_lr * 0.05
            num_step_in_epoch = self.state['num_train_batches']
            warmup_steps = num_step_in_epoch * config.lr_warmup_epochs
            max_steps = num_step_in_epoch * config.lr_decay_epochs

            # 1) linear warmup for warmup_iters steps
            if step < warmup_steps:
                return max_lr * (step + 1) / warmup_steps
            # 2) if it > lr_decay_iters, return min learning rate
            if step > max_steps:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
            return min_lr + coeff * (max_lr - min_lr)
        
        scheduler = optim.lr_scheduler.LambdaLR(self.state['optimizer'], lr_lambda=multiplicative_schedule)
        scheduler._step_count = -1 # To prevent warning because initialation makes a first call to step automatically
        return scheduler


@train_app.command("new")
def train(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        bs: int = typer.Option(32, min=1, help="Batch Size"),
        dim: int = typer.Option(128, min=8, max=512, help="Dimension of the model"),
        heads: int = typer.Option(16, min=1, max=64, help="Number of heads within each self-attention block"),
        blocks: int = typer.Option(3, min=1, max=20, help="Number of mixing blocks within each recurrent layer"),
        mixers: int = typer.Option(3, min=1, max=10, help="Number of mixers within each mixing block"),
        layers: int = typer.Option(3, min=1, max=10, help="Number of recurrent layers"),
        mlr: float = typer.Option(0.01, min=-1.0, help="Learning Rate. If -1, then learning rate finder is invoked in debug model."),
        plr: Optional[float] = typer.Option(None, min=-1.0, help="Program Learning Rate. If None, then it is scaled according to data augmentation level"),
        lr_warmup: int = typer.Option(2, min=0, help="Number of epochs for learning rate warmup"),
        lr_decay: int = typer.Option(8, min=0, help="Number of epochs for learning rate decay"),
        mwd: float = typer.Option(0.01, min=0.0, help="Weight Decay"),
        pwd: float = typer.Option(0.0, min=0.0, help="Program Weight Decay"),
        data_aug: int = typer.Option(3, min=0, help="Data Augmentation Level. 0 means no augmentation"),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment"),
        sep_task_version: bool = typer.Option(True, help="If set, task ID and task version are given separate embeddings"),
        share_mixer: bool = typer.Option(True, help="Share mixer within each mixing block"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        device: Optional[str] = typer.Option(None, help="Device to run the training on. If None, then it is automatically selected"),
        eval_int: Optional[int] = typer.Option(None, help="Number of steps between evaluations. None means evaluation at the end of each epoch"),
        num_max_examples: int = typer.Option(-1, min=-1, help="For test runs. Takes the subset of the training and evaluation data. value <= 0 means all data"),
    ):


    if plr is None:
        # There is factor of 8 samples for every level of data augmentation
        plr_scale = 1 if data_aug <= 0 else 8 * data_aug
        plr = mlr * plr_scale


    hparams = SolverHparams(experiment=name, run=run, seed=seed, device=device, eval_interval=eval_int)
    data_config = {
        "data_aug": data_aug,
        "sep_task_version": sep_task_version,
    }

    model_config = {
        "n_dim": dim,
        "n_heads": heads,
        "n_blocks": blocks,
        "n_mixers": mixers,
        "n_layers": layers,
        "share_mixer": share_mixer
    }

    optimizer_config = {
        "batch_size": bs,
        "model_lr": mlr if not lr_find else 1,
        "model_wd": mwd,
        "prog_lr": plr if not lr_find else plr_scale,
        "prog_wd": pwd,
        "lr_warmup_epochs": lr_warmup,
        "lr_decay_epochs": lr_decay,
        "max_examples": num_max_examples
    }

    
    hparams.add_params(prefix="data", **data_config)
    hparams.add_params(prefix="model", **model_config)
    hparams.add_params(prefix="optim", **optimizer_config)

    trainer = ArcTrainer(hparams=hparams)
    
    if lr_find:
        trainer.find_lr()
    else:
        trainer.train()


@train_app.command("resume")
def resume(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        checkpoint: Optional[str] = typer.Option(None, help="Name or number or path of the checkpoint to resume from. If None, then the latest checkpoint is used."),
    ):

    # ArcTrainer._resume(name, run, checkpoint)
    pass

@train_app.command("from")
def from_checkpoint(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        checkpoint: Optional[str] = typer.Option(None, help="Name or number or path of the checkpoint to resume from. If None, then the latest checkpoint is used."),
    ):
    pass



if __name__ == "__main__":
    app()
