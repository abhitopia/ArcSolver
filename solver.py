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

logger = get_logger()

_DEV_MODE = "abhishekaggarwal" in str(Path(__file__).resolve())
_BASE_DIR = "./runs"

if _DEV_MODE:
    logger.warning("WARNING: Running in DEV mode")

def train_from_hparams(hparams, checkpoint, lr_find, debug=False, parent_dir=_BASE_DIR):
    if lr_find or debug:
        hparams.run = f"{hparams.run}/debug"

    trainer = ArcTrainer(hparams=hparams,
                        parent_dir=parent_dir,
                        prevent_overwrite=True,
                        disable_checkpointing_and_logging=True if (lr_find or debug) else False)
    if checkpoint is not None:
        existing_checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)
        assert existing_checkpoint is None, f"Checkpoint {existing_checkpoint} already exists. Loading from checkpoint will overwrite the existing checkpoint"
        trainer.initialise_from_checkpoint(checkpoint)    # NO RESUME, start from the beginning 

    if lr_find:
        trainer.find_lr()
    else:
        trainer.train()

def get_checkpoint(name, run, checkpoint=None, parent_dir=_BASE_DIR):
    checkpoint_dir = ArcTrainer.get_checkpoint_dir(name, run, parent_dir=parent_dir)
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"

    if checkpoint is None:
        checkpoint = ArcTrainer.get_latest_checkpoint(checkpoint_dir)
        assert checkpoint is not None, f"No checkpoint found in {checkpoint_dir}"

    checkpoint = Path(checkpoint)
    assert checkpoint.parent == checkpoint_dir, f"Checkpoint {checkpoint} is not in the checkpoint directory {checkpoint_dir}"
    return checkpoint

def split_run_path(run_path: str) -> Tuple[str, str]:
    run_path = Path(run_path)
    assert run_path.exists(), f"Path {run_path} does not exist"
    assert run_path.is_dir(), f"Path {run_path} is not a directory"
    run = run_path.name
    experiment = run_path.parent.name
    parent_dir = run_path.parent.parent
    return experiment, run, parent_dir

class LRSchedule(str, Enum):
    noam = "noam"
    alt = "alt"
    const = "const"

@train_app.command("new")
def train(
        experiment: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        bs: int = typer.Option(32, min=1, help="Batch Size"),
        prog_dim: int = typer.Option(4, min=4, max=512, help="Dimension of the model"),
        heads: int = typer.Option(4, min=1, max=64, help="Number of heads within each self-attention block"),
        blocks: int = typer.Option(1, min=1, max=20, help="Number of mixing blocks within each recurrent layer"),
        n_rec_block: int = typer.Option(1, min=1, max=10, help="Block level recurrence"),
        n_rec_layer: int = typer.Option(1, min=1, max=10, help="Layer level recurrence"),
        mlr: float = typer.Option(0.001, min=-1.0, help="Learning Rate"),
        plr: Optional[float] = typer.Option(None, min=0.0, help="Program Learning Rate. If None, then it is automatically determined based on the schedule and data augmentation"),
        lr_warmup: int = typer.Option(2, min=0, help="Number of epochs for learning rate warmup. Only used for noam scheduler"),
        lr_decay: int = typer.Option(8, min=0, help="Number of epochs for learning rate decay. Only used for noam scheduler"),
        lr_schedule: LRSchedule = typer.Option(LRSchedule.noam, help="Learning rate scheduler. Options: noam, alt, const"),
        mwd: float = typer.Option(0.01, min=0.0, help="Weight Decay"),
        pwd: float = typer.Option(0.0, min=0.0, help="Program Weight Decay"),
        data_aug: int = typer.Option(3, min=0, help="Data Augmentation Level. 0 means no augmentation"),
        num_diff_levels: int = typer.Option(15, min=1, help="Number of partitions of the data based on difficulty"),
        diff_level: int = typer.Option(15, min=1, help="Difficulty level of the training data. Must be less than or equal to num_diff_levels"),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        device: Optional[str] = typer.Option(None, help="Device to run the training on. If None, then it is automatically selected"),
        eval_int: Optional[int] = typer.Option(None, help="Number of steps between evaluations. None means evaluation at the end of each epoch"),
        debug: Optional[bool] = typer.Option(False, help="For test runs. Nothing is saved"),
        checkpoint: Optional[str] = typer.Option(None, help="Initialize the model from the given checkpoint. Training will start from the beginning")
    ):

    hparams = ArcHparams(experiment=experiment, run=run, seed=seed, device=device, eval_interval=eval_int)
    data_config = {
        "data_aug": data_aug,
        "diff_level": diff_level,
        "num_diff_levels": num_diff_levels
    }

    model_config = {
        "n_prog_embd": prog_dim,
        "n_heads": heads,
        "n_blocks": blocks,
        "n_rec_block": n_rec_block,
        "n_rec_layer": n_rec_layer,
    }

    optimizer_config = {
        "batch_size": bs,  # Yes, this is optimizer config
        "lr_model": mlr if not lr_find else 1,
        "wd_model": mwd,
        "lr_prog": plr if not lr_find else 1,
        "wd_prog": pwd,
        "lr_schedule": lr_schedule.value,
        "lr_warmup_epochs": lr_warmup,
        "lr_decay_epochs": lr_decay,
        "max_examples": 1000 if _DEV_MODE else -1 # Yes, this is optimizer config
    }

    hparams.add_params(prefix="data", **data_config)
    hparams.add_params(prefix="model", **model_config)
    hparams.add_params(prefix="optim", **optimizer_config)
    train_from_hparams(hparams, checkpoint, lr_find, debug)


@train_app.command("resume")
def resume(
       run_path: str = typer.Argument(..., help="Path to the run folder (not checkpoint) to resume training from"),
    ):

    experiment, run, parent_dir = split_run_path(run_path)
    checkpoint_dir = ArcTrainer.get_checkpoint_dir(experiment, run, parent_dir=parent_dir)
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
    checkpoint = ArcTrainer.get_latest_checkpoint(checkpoint_dir)       
    trainer = ArcTrainer.from_checkpoint(checkpoint, resume=True)
    trainer.train()


@train_app.command("info")
def info(
       run_path: str = typer.Argument(..., help="Path to the run folder (not checkpoint) to resume training from"),
    ):
    experiment, run, parent_dir = split_run_path(run_path)
    checkpoint_dir = ArcTrainer.get_checkpoint_dir(experiment, run, parent_dir=parent_dir)
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
    checkpoint = ArcTrainer.get_latest_checkpoint(checkpoint_dir)       
    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    print(hparams_dict)

def get_run_name(hparams_dict, new_hparams_dict):
    diff_dict = get_diff_dict(hparams_dict, new_hparams_dict)
    assert len(diff_dict) > 0, "At least one parameter should be changed"
    new_run = hparams_dict['run'] 
    for key, value in diff_dict.items():
        name_key = key.split("_")[-1]
        print(name_key, key)
        if key == "run":
            continue
        new_run += f"|{name_key}_{new_hparams_dict[key]}"
    return new_run

@change_app.command("optim")
def optim_change(
        run_path: str = typer.Argument(..., help="Path to the run folder (not checkpoint) to resume training from"),
        mlr: float = typer.Option(None, min=0.0, help="Model Learning Rate"),
        plr: Optional[float] = typer.Option(None, min=0.0, help="Program Learning Rate. If None, then it is scaled according to data augmentation level"),
        lr_schedule: LRSchedule = typer.Option(None, help="Learning rate scheduler. Options: noam, alt, const"),
        bs: int = typer.Option(None, min=1, help="Batch Size"),
        lr_warmup: int = typer.Option(None, min=0, help="Number of epochs for learning rate warmup. Only used for noam scheduler"),
        lr_decay: int = typer.Option(None, min=0, help="Number of epochs for learning rate decay. Only used for noam scheduler"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        checkpoint: Optional[str] = typer.Option(None, help="Use this specific checkpoint"),
        debug: Optional[bool] = typer.Option(False, help="For test runs. Nothing is saved")
    ):
    experiment, run, parent_dir = split_run_path(run_path)
    checkpoint = get_checkpoint(experiment, run, checkpoint, parent_dir=parent_dir)
    logger.info(f"Base Checkpoint: {checkpoint}")

    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    logger.info(f"Base Hparams: {hparams_dict}")

    update_dict =  {
        "batch_size": bs if bs is not None else hparams_dict['optim.batch_size'],  # Yes, this is optimizer config
        "lr_model": 1 if  lr_find else (mlr if mlr is not None else hparams_dict['optim.lr_model']),
        "lr_prog": plr if not lr_find else 1,
        "lr_schedule": lr_schedule.value if lr_schedule is not None else hparams_dict['optim.lr_schedule'],
        "lr_warmup_epochs": lr_warmup if lr_warmup is not None else hparams_dict['optim.lr_warmup_epochs'],
        "lr_decay_epochs": lr_decay if lr_decay is not None else hparams_dict['optim.lr_decay_epochs'],
    }

    update_dict = {f'optim.{k}': v for k, v in update_dict.items()}
    new_hparams_dict = deepcopy(hparams_dict)
    new_hparams_dict.update(update_dict)
    new_run = get_run_name(hparams_dict, new_hparams_dict)
    new_hparams_dict['run'] = new_run

    logger.info(f"Changed Hparams: {get_diff_dict(hparams_dict, new_hparams_dict)}")
    new_params = ArcHparams.from_dict(new_hparams_dict)
    train_from_hparams(new_params, checkpoint, lr_find, debug, parent_dir=parent_dir)



@change_app.command("model")
def change_model_size(
        run_path: str = typer.Argument(..., help="Path to the run folder (not checkpoint) to resume training from"),
        dim: int = typer.Option(None, min=8, max=512, help="Dimension of the model"),
        heads: int = typer.Option(None, min=1, max=64, help="Number of heads within each self-attention block"),
        blocks: int = typer.Option(None, min=1, max=20, help="Number of mixing blocks within each recurrent layer"),
        mixers: int = typer.Option(None, min=1, max=10, help="Number of mixers within each mixing block"),
        layers: int = typer.Option(None, min=1, max=10, help="Number of recurrent layers"),
        share_mixer: bool = typer.Option(True, help="Share mixer within each mixing block"),
        checkpoint: Optional[str] = typer.Option(None, help="Use this specific checkpoint"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        debug: Optional[bool] = typer.Option(False, help="For test runs. Nothing is saved")
    ):

    experiment, run, parent_dir = split_run_path(run_path)
    checkpoint = get_checkpoint(experiment, run, checkpoint, parent_dir=parent_dir)
    logger.info(f"Base Checkpoint: {checkpoint}")

    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    logger.info(f"Base Hparams: {hparams_dict}")


    update_dict = {
        "n_dim": dim if dim is not None else hparams_dict['model.n_dim'],
        "n_heads": heads if heads is not None else hparams_dict['model.n_heads'],
        "n_blocks": blocks if blocks is not None else hparams_dict['model.n_blocks'],
        "n_mixers": mixers if mixers is not None else hparams_dict['model.n_mixers'],
        "n_layers": layers if layers is not None else hparams_dict['model.n_layers'],
        "share_mixer": share_mixer
    }

    update_dict = {f'model.{k}': v for k, v in update_dict.items()}    
    new_hparams_dict = deepcopy(hparams_dict)
    new_hparams_dict.update(update_dict)
    new_run = get_run_name(hparams_dict, new_hparams_dict)
    new_hparams_dict['run'] = new_run
    logger.info(f"Changed Hparams: {get_diff_dict(hparams_dict, new_hparams_dict)}")
    new_params = ArcHparams.from_dict(new_hparams_dict)
    train_from_hparams(new_params, checkpoint, lr_find, debug, parent_dir=parent_dir)



if __name__ == "__main__":
    app()
