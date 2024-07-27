from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import typer


from src.arc_trainer import ArcTrainer, ArcHparams
from rich import print

from src.utils import get_diff_dict, get_logger

app = typer.Typer(pretty_exceptions_show_locals=False)
train_app = typer.Typer()
change_app = typer.Typer()
lr_app = typer.Typer()
app.add_typer(train_app, name="train")
train_app.add_typer(change_app, name="change")
change_app.add_typer(lr_app, name="lr")


logger = get_logger()

_DEV_MODE = True
# _BASE_DIR = "./runs"
_BASE_DIR = "./lightning_runs/runs"

class LRSchedule(str, Enum):
    noam = "noam"
    alt = "alt"
    const = "const"


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
        mlr: float = typer.Option(0.001, min=-1.0, help="Learning Rate"),
        plr: Optional[float] = typer.Option(None, min=0.0, help="Program Learning Rate. If None, then it is automatically determined based on the schedule and data augmentation"),
        lr_warmup: int = typer.Option(2, min=0, help="Number of epochs for learning rate warmup. Only used for noam scheduler"),
        lr_decay: int = typer.Option(8, min=0, help="Number of epochs for learning rate decay. Only used for noam scheduler"),
        lr_schedule: LRSchedule = typer.Option(LRSchedule.noam, help="Learning rate scheduler. Options: noam, alt, const"),
        mwd: float = typer.Option(0.01, min=0.0, help="Weight Decay"),
        pwd: float = typer.Option(0.0, min=0.0, help="Program Weight Decay"),
        data_aug: int = typer.Option(3, min=0, help="Data Augmentation Level. 0 means no augmentation"),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment"),
        sep_task_version: bool = typer.Option(True, help="If set, task ID and task version are given separate embeddings"),
        share_mixer: bool = typer.Option(True, help="Share mixer within each mixing block"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        device: Optional[str] = typer.Option(None, help="Device to run the training on. If None, then it is automatically selected"),
        eval_int: Optional[int] = typer.Option(None, help="Number of steps between evaluations. None means evaluation at the end of each epoch"),
        debug: Optional[bool] = typer.Option(False, help="For test runs. Nothing is saved"),
        checkpoint: Optional[str] = typer.Option(None, help="Initialize the model from the given checkpoint. Training will start from the beginning")
    ):


    if lr_find or debug:
        run = f"{run}/debug"

    hparams = ArcHparams(experiment=name, run=run, seed=seed, device=device, eval_interval=eval_int)
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
        "batch_size": bs,  # Yes, this is optimizer config
        "model_lr": mlr if not lr_find else 1,
        "model_wd": mwd,
        "prog_lr": plr if not lr_find else 1,
        "prog_wd": pwd,
        "lr_schedule": lr_schedule.value,
        "lr_warmup_epochs": lr_warmup,
        "lr_decay_epochs": lr_decay,
        "max_examples": 1000 if _DEV_MODE else None # Yes, this is optimizer config
    }

    
    hparams.add_params(prefix="data", **data_config)
    hparams.add_params(prefix="model", **model_config)
    hparams.add_params(prefix="optim", **optimizer_config)

    trainer = ArcTrainer(hparams=hparams,
                        parent_dir=_BASE_DIR,
                        disable_checkpointing_and_logging=True if (lr_find or debug) else False)
    if checkpoint is not None:
        existing_checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)
        assert existing_checkpoint is None, f"Checkpoint {existing_checkpoint} already exists. Loading from checkpoint will overwrite the existing checkpoint"
        trainer.initialise_from_checkpoint(checkpoint)    # NO RESUME, start from the beginning 

    if lr_find:
        trainer.find_lr()
    else:
        trainer.train()


@train_app.command("resume")
def resume(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
    ):
    
    checkpoint_dir = ArcTrainer.get_checkpoint_dir(name, run, parent_dir=_BASE_DIR)
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
    checkpoint = ArcTrainer.get_latest_checkpoint(checkpoint_dir)       
    trainer = ArcTrainer.from_checkpoint(checkpoint, resume=True)
    trainer.train()


@train_app.command("info")
def info(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
    ):
    ## Get the info on the latest checkpoint. Show the hparams

    checkpoint_dir = ArcTrainer.get_checkpoint_dir(name, run, parent_dir=_BASE_DIR)
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
    checkpoint = ArcTrainer.get_latest_checkpoint(checkpoint_dir)       
    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    print(hparams_dict)


def get_checkpoint(name, run, checkpoint):
    checkpoint_dir = ArcTrainer.get_checkpoint_dir(name, run, parent_dir=_BASE_DIR)
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"

    if checkpoint is None:
        checkpoint = ArcTrainer.get_latest_checkpoint(checkpoint_dir)
        assert checkpoint is not None, f"No checkpoint found in {checkpoint_dir}"

    checkpoint = Path(checkpoint)
    assert checkpoint.parent == checkpoint_dir, f"Checkpoint {checkpoint} is not in the checkpoint directory {checkpoint_dir}"
    return checkpoint

def change_hparams_and_train(checkpoint, hparams_dict, update_dict, lr_find):
    new_hparams_dict = deepcopy(hparams_dict)
    new_hparams_dict.update(update_dict)
    logger.info(f"Changed Hparams: {get_diff_dict(hparams_dict, new_hparams_dict)}")
    

    new_params = ArcHparams.from_dict(new_hparams_dict)
    if lr_find:
        new_params.run = new_params.run + "/debug"

    trainer = ArcTrainer(hparams=new_params,
                        parent_dir=_BASE_DIR,
                        disable_checkpointing_and_logging=True if lr_find else False)
    
    trainer.initialise_from_checkpoint(checkpoint, strict=False)    # NO RESUME, start from the beginning 
    if lr_find:
        trainer.find_lr()
    else:
        trainer.train()



@change_app.command("lr")
def lr_change(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        mlr: float = typer.Argument(..., min=-1.0, help="Model Learning Rate"),
        plr: Optional[float] = typer.Option(None, min=0.0, help="Program Learning Rate. If None, then it is scaled according to data augmentation level"),
        lr_schedule: LRSchedule = typer.Option(LRSchedule.noam, help="Learning rate scheduler. Options: noam, alt, const"),
        lr_warmup: int = typer.Option(2, min=0, help="Number of epochs for learning rate warmup. Only used for noam scheduler"),
        lr_decay: int = typer.Option(8, min=0, help="Number of epochs for learning rate decay. Only used for noam scheduler"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode")
    ):
    ## Change the learning rate of the model and the program
    
    checkpoint = get_checkpoint(name, run, checkpoint)
    logger.info(f"Base Checkpoint: {checkpoint}")

    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    logger.info(f"Base Hparams: {hparams_dict}")

    update_dict =  {
        "model_lr": mlr if not lr_find else 1,
        "prog_lr": plr if not lr_find else 1,
        "lr_schedule": lr_schedule.value,
        "lr_warmup_epochs": lr_warmup,
        "lr_decay_epochs": lr_decay,
    }

    change_hparams_and_train(checkpoint, hparams_dict, update_dict, lr_find)



class ModelSize(str, Enum):
    mixers = "mixers"
    blocks = "blocks"
    layers = "layers"

    @property
    def param(self):
        return {
            ModelSize.mixers.value: "model.n_mixers",
            ModelSize.blocks.value: "model.n_blocks",
            ModelSize.layers.value: "model.n_layers"
        }[self.value]


@change_app.command("model")
def change_model_size(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        key: ModelSize = typer.Argument(ModelSize.mixers, help="Key to change"),
        value: int = typer.Argument(..., help="Number of mixers within each mixing block"),
        checkpoint: Optional[str] = typer.Option(None, help="Use this specific checkpoint"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode")
    ):

    checkpoint = get_checkpoint(name, run, checkpoint)
    logger.info(f"Base Checkpoint: {checkpoint}")

    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    logger.info(f"Base Hparams: {hparams_dict}")

    update_dict = {
                key.param: value,
                'run': hparams_dict['run'] + f"_{key.name[0]}{value}"
                }
    change_hparams_and_train(checkpoint, hparams_dict, update_dict, lr_find)



if __name__ == "__main__":
    app()
