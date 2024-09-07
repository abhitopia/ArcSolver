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

        # Batch Config
        bs: int = typer.Option(32, min=1, help="Batch Size"),
        bsl: int = typer.Option(1024, min=1, help="Batch Seq Length"),
        dynamic_batching: bool = typer.Option(True, help="Use dynamic batch size"),

        # Model Config
        n_dim: int = typer.Option(16, min=4, max=512, help="Dimension of the model"),
        heads: int = typer.Option(4, min=1, max=64, help="Number of heads within each self-attention block"),
        blocks: int = typer.Option(1, min=1, max=20, help="Number of mixing blocks within each recurrent layer"),
        dropout: float = typer.Option(0.0, min=0.0, max=1.0, help="Dropout probability"),

        # Loop Config
        max_loops: int = typer.Option(16, min=1, max=100, help="Network level recurrence"),
        min_loops: int = typer.Option(2, min=0, help="Minimum number of loops"),
        inc_loops: int = typer.Option(1, min=1, help="Number of loops to increase by"),
        int_loops: int = typer.Option(100, min=1, help="Interval for increasing number of loops"),
        max_loops_prob: float = typer.Option(0.5, min=0.0, max=1.0, help="Probability of choosing max loops during training"),
        start_loops: Optional[int] = typer.Option(None, min=1, help="Starting number of loops for the curriculum"),

        # Learning Rate Config
        mlr: float = typer.Option(0.001, min=-1.0, help="Learning Rate"),
        plr: Optional[float] = typer.Option(None, min=0.0, help="Program Learning Rate. If None, then it is set to mlr"),
        lr_warmup: int = typer.Option(100, min=0, help="Number of steps for learning rate warmup. Only used for noam and lindecay scheduler"),
        lr_decay: int = typer.Option(1000, min=0, help="Number of steps for learning rate decay. Only used for noam and lindecay scheduler"),
        lr_schedule: LRSchedule = typer.Option(LRSchedule.noam, help="Learning rate scheduler. Options: noam, alt, const"),

        # Weight Decay Config
        mwd: float = typer.Option(0.01, min=0.0, help="Weight Decay"),
        pwd: float = typer.Option(0.0, min=0.0, help="Program Weight Decay"),

        # Grok Config
        grok_alpha: float = typer.Option(0.0, min=0.0, help="Grok Alpha"),
        grok_lambda: float = typer.Option(0.0, min=0.0, help="Grok Lambda"),

        # Data Config
        data_aug: int = typer.Option(3, min=0, help="Data Augmentation Level. 0 means no augmentation"),
        num_diff_levels: int = typer.Option(5, min=1, help="Number of partitions of the data based on difficulty"),
        diff_level: int = typer.Option(1, min=1, help="Difficulty level of the training data. Must be less than or equal to num_diff_levels"),
        use_aux: bool = typer.Option(True, help="Use auxiliary data for training"),

        # Misc Config
        n_steps: Optional[int] = typer.Option(None, min=1, help="Number of steps to train for. If None, lr_decay + lr_warmup is used"),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        device: Optional[str] = typer.Option(None, help="Device to run the training on. If None, then it is automatically selected"),
        eval_int: Optional[int] = typer.Option(None, help="Number of steps between evaluations. None means evaluation at the end of each epoch"),
        debug: Optional[bool] = typer.Option(False, help="For test runs. Nothing is saved"),

        # Checkpoint Config
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
        "dropout": dropout
    }

    optimizer_config = {
        # Batch Size
        "batch_size": bs,  # Yes, this is optimizer config
        "batch_seq_len": bsl,
        "dynamic_batching": dynamic_batching,

        # Loop Curriculum
        "max_loops": max_loops,
        "min_loops": min_loops,
        "inc_loops": inc_loops,
        "int_loops": int_loops,
        "max_loops_prob": max_loops_prob,
        "start_loops": start_loops if start_loops is not None else min_loops,

        # Learning Rate
        "lr_model": mlr if not lr_find else 1,
        "lr_prog": plr if not lr_find else 1,
        "lr_schedule": lr_schedule.value if isinstance(lr_schedule, LRSchedule) else lr_schedule,
        "lr_warmup_steps": lr_warmup,
        "lr_decay_steps": lr_decay,

        # Weight Decay
        "wd_model": mwd,
        "wd_prog": pwd,
        
        "max_examples": 5000 if _DEV_MODE else -1 # Yes, this is optimizer config
    }

    hparams.add_params(prefix="data", **data_config)
    hparams.add_params(prefix="model", **model_config)
    hparams.add_params(prefix="optim", **optimizer_config)


    if debug:
        hparams.run = f"debug_{hparams.run}"

    trainer = ArcTrainer(hparams=hparams,
                        parent_dir=_BASE_DIR,
                        prevent_overwrite=True,
                        num_checkpoints_to_keep=4 if hparams.run == run else 3,
                        disable_checkpointing_and_logging=True if (lr_find or debug) else False)
    
    if checkpoint is not None:
        existing_checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)
        assert existing_checkpoint is None, f"Checkpoint {existing_checkpoint} already exists. Loading from checkpoint will overwrite the existing checkpoint"
        trainer.initialise_from_checkpoint(checkpoint, strict=False, load_model=True, load_optim=False)    # NO RESUME, start from the beginning 

    if lr_find:
        trainer.find_lr()
    else:
        trainer.train(max_steps=n_steps if n_steps is not None else hparams.optim.lr_decay_steps + hparams.optim.lr_warmup_steps)



@train_app.command("fork")
def fork(
        experiment: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        checkpoint: str = typer.Argument(..., help="Path to the checkpoint file to fork training from"),

        # Misc Config
        n_steps: Optional[int] = typer.Option(None, min=1, help="Number of steps to train for. If None, lr_decay + lr_warmup is used"),
        seed: Optional[int] = typer.Option(None, min=0, help="Random seed for the data and experiment"),


        # Loop Curriculum
        start_loops: Optional[int] = typer.Option(None, min=1, help="Starting number of loops for the curriculum"),
        max_loops: Optional[int] = typer.Option(None, min=1, max=100, help="Network level recurrence"),
        inc_loops: int = typer.Option(1, min=1, help="Number of loops to increase by"),
        int_loops: int = typer.Option(100, min=1, help="Interval for increasing number of loops"),
        min_loops: int = typer.Option(2, min=0, help="Minimum number of loops"),
        max_loops_prob: float = typer.Option(0.5, min=0.0, max=1.0, help="Probability of choosing max loops during training"),

        # Batch Config
        bs: Optional[int] = typer.Option(None, min=1, help="Batch Size"),
        bsl: Optional[int] = typer.Option(None, min=1, help="Batch Seq Length"),
        dynamic_batching: Optional[bool] = typer.Option(True, help="Use dynamic batch size"),

        # Learning Rate Config
        mlr: Optional[float] = typer.Option(None, min=0.0, help="Learning Rate"),
        plr: Optional[float] = typer.Option(None, min=0.0, help="Program Learning Rate. If None, then it is set to mlr"),
        lr_warmup: Optional[int] = typer.Option(None, min=0, help="Number of steps for learning rate warmup. Only used for noam and lindecay scheduler"),
        lr_decay: Optional[int] = typer.Option(None, min=0, help="Number of steps for learning rate decay. Only used for noam and lindecay scheduler"),
        lr_schedule: Optional[LRSchedule] = typer.Option(None, help="Learning rate scheduler. Options: noam, alt, const"),


        # Weight Decay Config
        mwd: Optional[float] = typer.Option(None, min=0.0, help="Weight Decay"),
        pwd: Optional[float] = typer.Option(None, min=0.0, help="Program Weight Decay"),

        # Model Config
        dropout: Optional[float] = typer.Option(None, min=0.0, max=1.0, help="Dropout probability"),

        # Grok Config
        grok_alpha: Optional[float] = typer.Option(None, min=0.0, help="Grok Alpha"),
        grok_lambda: Optional[float] = typer.Option(None, min=0.0, help="Grok Lambda"),

        # Data Config
        data_aug: Optional[int] = typer.Option(None, min=0, help="Data Augmentation Level. 0 means no augmentation"),
        num_diff_levels: Optional[int] = typer.Option(None, min=1, help="Number of partitions of the data based on difficulty"),
        diff_level: Optional[int] = typer.Option(None, min=1, help="Difficulty level of the training data. Must be less than or equal to num_diff_levels"),
        use_aux: Optional[bool] = typer.Option(True, help="Use auxiliary data for training"),

        # Misc Config
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        debug: Optional[bool] = typer.Option(False, help="For test runs. Nothing is saved"),
    ):

    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    hparams = ArcHparams.from_dict(hparams_dict)

    prev_exp = f"{hparams.experiment}_{hparams.run}"
    new_exp = f"{experiment}_{run}"

    is_resume = prev_exp == new_exp

    base_config = {
        "experiment": experiment,
        "run": run,
        "seed": seed,
        "grok_alpha": grok_alpha,
        "grok_lambda": grok_lambda,
    }

    data_config = {
        "data_aug": data_aug,
        "diff_level": diff_level,
        "num_diff_levels": num_diff_levels,
        "use_aux": use_aux
    }

    model_config = {
        "dropout": dropout
    }

    optimizer_config = {
        # Batch Size
        "batch_size": bs,  # Yes, this is optimizer config
        "batch_seq_len": bsl,
        "dynamic_batching": dynamic_batching,

        # Loop Curriculum
        "max_loops": max_loops,
        "min_loops": min_loops,
        "inc_loops": inc_loops,
        "int_loops": int_loops,
        "max_loops_prob": max_loops_prob,
        "start_loops": start_loops if start_loops is not None else min_loops,

        # Learning Rate
        "lr_model": mlr if not lr_find else 1,
        "lr_prog": plr if not lr_find else 1,
        "lr_schedule": lr_schedule.value if isinstance(lr_schedule, LRSchedule) else lr_schedule,
        "lr_warmup_steps": lr_warmup,
        "lr_decay_steps": lr_decay,

        # Weight Decay
        "wd_model": mwd,
        "wd_prog": pwd,        
    }

    def override_hparams(hparams, config):
        for key, value in config.items():
            if value is not None:
                setattr(hparams, key, value)


    override_hparams(hparams, base_config)
    override_hparams(hparams.data, data_config)
    override_hparams(hparams.model, model_config)
    override_hparams(hparams.optim, optimizer_config)


    if debug:
        hparams.run = f"debug_{hparams.run}"

    trainer = ArcTrainer(hparams=hparams,
                        parent_dir=_BASE_DIR,
                        prevent_overwrite=False if is_resume else True, # Important if resuming
                        disable_checkpointing_and_logging=True if (lr_find or debug) else False)


    existing_checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)

    if not is_resume:
        existing_checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)
        assert existing_checkpoint is None, f"Checkpoint {existing_checkpoint} already exists. Loading from checkpoint will overwrite the existing checkpoint"
    else:
        trainer.info("Attemping to resume {experiment}/{run}")

    trainer.initialise_from_checkpoint(checkpoint, strict=False, load_model=True, load_optim=True)    # Fork start from the beginning 

    if lr_find:
        trainer.find_lr()
    else:
        trainer.train(max_steps=n_steps if n_steps is not None else hparams.optim.lr_decay_steps + hparams.optim.lr_warmup_steps)


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
