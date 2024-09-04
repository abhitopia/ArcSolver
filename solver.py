from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import typer
from src.arc_trainer import ArcTrainer, ArcHparams
from rich import print
from src.utils import generate_random_sweep_config, get_logger, construct_sweep_config

app = typer.Typer(pretty_exceptions_show_locals=False)
train_app = typer.Typer()
change_app = typer.Typer()
app.add_typer(train_app, name="train")

logger = get_logger()

_DEV_MODE = "abhishekaggarwal" in str(Path(__file__).resolve())
_BASE_DIR = "./runs"

if _DEV_MODE:
    logger.warning("WARNING: Running in DEV mode")

def train_from_hparams(hparams, checkpoint, lr_find, n_steps, debug=False, parent_dir=_BASE_DIR):
    if lr_find or debug:
        hparams.run = f"debug_{hparams.run}"

    trainer = ArcTrainer(hparams=hparams,
                        parent_dir=parent_dir,
                        prevent_overwrite=True,
                        disable_checkpointing_and_logging=True if (lr_find or debug) else False)
    if checkpoint is not None:
        existing_checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)
        assert existing_checkpoint is None, f"Checkpoint {existing_checkpoint} already exists. Loading from checkpoint will overwrite the existing checkpoint"
        trainer.initialise_from_checkpoint(checkpoint, strict=False)    # NO RESUME, start from the beginning 

    if lr_find:
        trainer.find_lr()
    else:
        trainer.train(max_steps=n_steps if n_steps is not None else hparams.optim.lr_decay_steps + hparams.optim.lr_warmup_steps)

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
    lindecay = "lindecay"

@train_app.command("new")
def train(
        experiment: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        bs: int = typer.Option(32, min=1, help="Batch Size"),
        n_dim: int = typer.Option(16, min=4, max=512, help="Dimension of the model"),
        heads: int = typer.Option(4, min=1, max=64, help="Number of heads within each self-attention block"),
        blocks: int = typer.Option(1, min=1, max=20, help="Number of mixing blocks within each recurrent layer"),
        loops: int = typer.Option(1, min=1, max=100, help="Network level recurrence"),
        dropout: float = typer.Option(0.0, min=0.0, max=1.0, help="Dropout probability"),
        mlr: float = typer.Option(0.001, min=-1.0, help="Learning Rate"),
        plr: Optional[float] = typer.Option(None, min=0.0, help="Program Learning Rate. If None, then it is set to mlr"),
        lr_warmup: int = typer.Option(100, min=0, help="Number of steps for learning rate warmup. Only used for noam and lindecay scheduler"),
        lr_decay: int = typer.Option(1000, min=0, help="Number of steps for learning rate decay. Only used for noam and lindecay scheduler"),
        n_steps: Optional[int] = typer.Option(None, min=1, help="Number of steps to train for. If None, lr_decay + lr_warmup is used"),
        lr_schedule: LRSchedule = typer.Option(LRSchedule.noam, help="Learning rate scheduler. Options: noam, alt, const"),
        mwd: float = typer.Option(0.01, min=0.0, help="Weight Decay"),
        pwd: float = typer.Option(0.0, min=0.0, help="Program Weight Decay"),
        grok_alpha: float = typer.Option(0.0, min=0.0, help="Grok Alpha"),
        grok_lambda: float = typer.Option(0.0, min=0.0, help="Grok Lambda"),
        data_aug: int = typer.Option(3, min=0, help="Data Augmentation Level. 0 means no augmentation"),
        num_diff_levels: int = typer.Option(5, min=1, help="Number of partitions of the data based on difficulty"),
        diff_level: int = typer.Option(1, min=1, help="Difficulty level of the training data. Must be less than or equal to num_diff_levels"),
        use_aux: bool = typer.Option(True, help="Use auxiliary data for training"),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        bsl: int = typer.Option(1024, min=1, help="Batch Seq Length"),
        dynamic_batching: bool = typer.Option(True, help="Use dynamic batch size"),
        device: Optional[str] = typer.Option(None, help="Device to run the training on. If None, then it is automatically selected"),
        eval_int: Optional[int] = typer.Option(None, help="Number of steps between evaluations. None means evaluation at the end of each epoch"),
        debug: Optional[bool] = typer.Option(False, help="For test runs. Nothing is saved"),
        checkpoint: Optional[str] = typer.Option(None, help="Initialize the model from the given checkpoint. Training will start from the beginning")
    ):

    if _DEV_MODE:
        experiment = f"dev_{experiment}"

    hparams = ArcHparams(experiment=experiment,
                        run=run, 
                        seed=seed, 
                        device=device, 
                        eval_interval=eval_int,
                        grok_alpha=grok_alpha,
                        grok_lambda=grok_lambda)
    data_config = {
        "data_aug": data_aug,
        "diff_level": diff_level,
        "num_diff_levels": num_diff_levels,
        "use_aux": use_aux
    }

    model_config = {
        "n_dim": n_dim,
        "n_heads": heads,
        "n_layers": blocks,
        # "n_loops": loops,
        "dropout": dropout
    }

    optimizer_config = {
        "batch_size": bs,  # Yes, this is optimizer config
        "batch_seq_len": bsl,
        "dynamic_batching": dynamic_batching,
        "lr_model": mlr if not lr_find else 1,
        "wd_model": mwd,
        "lr_prog": plr if not lr_find else 1,
        "wd_prog": pwd,
        "lr_schedule": lr_schedule.value if isinstance(lr_schedule, LRSchedule) else lr_schedule,
        "lr_warmup_steps": lr_warmup,
        "lr_decay_steps": lr_decay,
        "max_examples": 5000 if _DEV_MODE else -1 # Yes, this is optimizer config
    }

    hparams.add_params(prefix="data", **data_config)
    hparams.add_params(prefix="model", **model_config)
    hparams.add_params(prefix="optim", **optimizer_config)
    train_from_hparams(hparams=hparams, checkpoint=checkpoint, lr_find=lr_find, n_steps=n_steps, debug=debug)


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


@train_app.command("random")
def random_sweep(
        experiment: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        prog_dim: int = typer.Option(16, min=4, max=512, help="Dimension of the model"),
        diff_level: int = typer.Option(5, min=1, help="Difficulty level of the training data. Must be less than or equal to num_diff_levels"),
        bsl: int = typer.Option(128, min=1, help="Batch Seq Length. BS is chosen randomly from [16, 32, 64, 128, 256]"),
        lr_decay: Optional[int] = typer.Option(5000, min=1, help="Number of steps to train for."),
        count: Optional[int] = typer.Option(10, min=1, help="Number of sequential runs"),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment")
    ):


    run = 0
    while run < count:
        print(f"Starting Random Config: {run+1}/{count}")
        SWEEP_DICT = {
            "bs": [16, 32, 64, 128, 256],
            "prog_dim": 16,
            "heads": [8, 16],
            "blocks": [1, 3, 5],
            "loops": [1, 2, 3],
            "dropout": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5],
            "mlr": [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001],
            "plr": None,
            "lr_warmup": [10, 50],
            "lr_decay": [1000],
            "n_steps": None,
            "lr_schedule": ["noam", "const"],
            "mwd": [0.0, 0.1, 0.01, 0.001, 0.0001],
            "pwd": [0.0, 0.1, 0.01, 0.001, 0.0001],
            "grok_alpha": [0.8, 0.85, 0.9, 0.95, 0.99],
            "grok_lambda": [0.1, 0.5, 1, 2, 5],
            "data_aug": [0, 1],
            "num_diff_levels": 10,
            "diff_level": 5,
            "use_aux": True,
            "seed": 42,
            "lr_find": False,
            "bsl": 128,
            "device": None,
            "eval_int": None,
            "debug": False,
            "checkpoint": None
        }


        sweep_dict = construct_sweep_config(SWEEP_DICT, experiment, prog_dim, diff_level=diff_level, bsl=bsl, lr_decay=lr_decay, seed=seed)
        print("Using following Sweep Config:")
        print(sweep_dict)

        config = generate_random_sweep_config(sweep_dict=sweep_dict)
        print("Using following Random Hparam Config:")
        print(config)
        try:
            train(**config)
            print(f"Training for {run+1} finished")
            run += 1
        except Exception:
            print("Error encountered. Trying again!")
            continue



if __name__ == "__main__":
    app()
