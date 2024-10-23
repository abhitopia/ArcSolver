import os

# Set the environment variable
# This is important when dealing with dynamic batches so GPU doesn't run out of memory
# See more here https://pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import typer
from src.arc_trainer import ArcTrainer
from src.arc_hparams import ArcHparams
from rich import print
from src.utils import get_logger

app = typer.Typer(pretty_exceptions_show_locals=False)

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

@app.command("new")
def train(
        experiment: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),

        # Batch Config
        tbs: int = typer.Option(10_000, min=1, help="Train Batch Size (in tokens)"),
        ebs: Optional[int] = typer.Option(None, min=1, help="Eval Batch Size (in tokens)"),
        grad_accum: Optional[int] = typer.Option(1, min=1, help="Number of steps to accumulate gradients over."),

        # Model Config
        n_dim: int = typer.Option(128, min=4, max=1024, help="Dimension of the model"),
        n_embd: int = typer.Option(16, min=4, max=512, help="Embedding dimension"),
        n_head: int = typer.Option(4, min=1, max=64, help="Number of heads within each self-attention block"),
        n_layer: int = typer.Option(3, min=1, max=20, help="Number of blocks in the Interpreter"),
        pnorm: bool = typer.Option(False, help="Program Norm. If True, then enforce 1 norm on all the embeddings"),
        rbase: int = typer.Option(10_000, min=5000, help="Base for geometric progression in angle computation"),
        n_iter: int = typer.Option(8, min=1, help="Number of iterations for the model"),

        # Loss / Compute Config
        gamma: Optional[float] = typer.Option(2.0, min=1.0, help="Loss Gamma. Gamma == 1 is equivalent to Cross Entropy Loss"),
        lalpha: Optional[float] = typer.Option(0.5, min=0.0, max=0.5, help="Portion of inverse loss, 0 <= lalpha <= 0.5"),
        # edr: Optional[float] = typer.Option(-1, min=-1.0, max=1.0,  help="Rate of error decay gradient. -1 falls back to regular cross entropy if mctp is 0.0"),
        # mctp: Optional[float] = typer.Option(0.0, min=0.0, help="Error % aftr the first iteration."),

        # Learning Rate Config
        mlr: Optional[float] = typer.Option(0.001, min=-1.0, help="Learning Rate"),
        plr: Optional[float] = typer.Option(0.001, min=0.0, help="Program Learning Rate. If None, then it is set to mlr"),
        lr_warmup: int = typer.Option(3_000, min=0, help="Number of steps for learning rate warmup. Only used for noam and lindecay scheduler"),
        lr_min_scale: Optional[float] = typer.Option(0.0, min=0.0, help="Learning Rate reached after decay phase is obtained by scaling the max learning rate by this factor"),
        lr_decay: int = typer.Option(None, min=0, help="Number of steps for learning rate decay. If None, then it is set to n_steps - lr_warmup"),
        lr_schedule: LRSchedule = typer.Option(LRSchedule.noam, help="Learning rate scheduler. Options: noam, alt, const"),
        plt_patience: int = typer.Option(10, min=0, help="Patience in number of training epochs which don't improve"),
        plt_factor: float = typer.Option(0.8, min=0.0, help="Factor for plateau scheduler to reduce learning rate"),
        plt_warmup: int = typer.Option(0, min=0, help="Warm up in number of updates before Pleateau Scheduler starts tracking Plateau metric"),

        # Regularisation/ Weight Decay Config
        mwd: float = typer.Option(0.5, min=0.0, help="Weight Decay"),
        pwd: float = typer.Option(0.0, min=0.0, help="Program Weight Decay"),
        pl1: Optional[float] = typer.Option(0.0, min=0.0, help="Program L1 Regularization"),
        dropout: float = typer.Option(0.0, min=0.0, max=1.0, help="Dropout probability"),

        # Grok Config
        grok_alpha: float = typer.Option(0.0, min=0.0, help="Grok Alpha"),
        grok_lambda: float = typer.Option(0.0, min=0.0, help="Grok Lambda"),

        # Data Config
        min_train_pp: Optional[int] = typer.Option(50, help="Minimum number of Train Examples Per Program"),
        max_train_pp: Optional[int] = typer.Option(None, help="Maximum number of Train Examples Per Program"),
        min_test_pp: Optional[int] = typer.Option(1, help="Minimum number of Test Examples Per Program"),
        max_test_pp: Optional[int] = typer.Option(3, help="Maximum number of Test Examples Per Program"),
        permute: bool = typer.Option(True, help="Permute the training set for each batch"),

        # Dataset Config
        include_train: bool = typer.Option(True, help="Include training data for training"),
        include_aux: bool = typer.Option(True, help="Include auxiliary data for training"),
        include_eval: bool = typer.Option(True, help="Include evaluation data for training"),
        include_inv: bool = typer.Option(False, help="Include inverse data for training"),

        # Misc Config
        n_steps: Optional[int] = typer.Option(1000000, min=1, help="Number of steps to train for. If None, lr_decay + lr_warmup is used"),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        device: Optional[str] = typer.Option(None, help="Device to run the training on. If None, then it is automatically selected"),
        eval_int: Optional[int] = typer.Option(None, help="Number of steps between evaluations. None means evaluation at the end of each epoch"),
        clear_cache_interval: Optional[int] = typer.Option(100, help="Clear cache before training"),
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

                        # Track Metric of interest for checkpointing purposes
                        target_metric='ARC_EVAL/Accuracy(%)' if include_eval else 'ARC_TRAIN/Accuracy(%)',
                        target_metric_increases=True,
                        num_checkpoints_to_keep=10, # Just in case

                        # Plateau Metric is always this
                        plt_metric='SampleAcc(%)',
                        plt_metric_increases=True,
                        plt_warmup=plt_warmup,
                        plateau_patience=plt_patience,
                        accumulation_steps=grad_accum,
                        plateau_factor=plt_factor,
                        grok_alpha=grok_alpha,
                        grok_lambda=grok_lambda)
    
    assert include_eval or include_train, "At least one of include_eval or include_train must be True"

    data_config = {
        'include_train': include_train,
        'include_aux': include_aux,
        'include_eval': include_eval,
        'include_inv': include_inv,
        'min_train_pp': min_train_pp,
        'max_train_pp': max_train_pp if max_train_pp is not None else min_train_pp,
        'min_test_pp': min_test_pp,
        'max_test_pp': max_test_pp if max_test_pp is not None else min_test_pp,
        'permute': permute
    }

    model_config = {
        "n_dim": n_dim,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "n_iter": n_iter,
        "rbase": rbase,
        "pnorm": pnorm,
        "gamma": gamma,
        "lalpha": lalpha
    }

    optimizer_config = {
        # Batch Size
        "train_batch_token_count": tbs,
        "eval_batch_token_count": ebs if ebs is not None else tbs,

        # Regularization / Weight Decay
        "wd_model": mwd,
        "wd_prog": pwd,
        "dropout": dropout,
        "l1_prog": pl1,

        # Loss / Compute Config
        # "edr": edr,
        # "mctp": mctp,

        # Learning Rate
        "lr_model": mlr if not lr_find else (1 if mlr > 0 else 0),
        "lr_prog": plr if not lr_find else (1 if plr > 0 else 0),
        "lr_schedule": lr_schedule.value if isinstance(lr_schedule, LRSchedule) else lr_schedule,
        "lr_warmup_steps": lr_warmup,
        "lr_decay_steps": lr_decay if lr_decay is not None else n_steps - lr_warmup,
        "lr_min_scale": lr_min_scale,
        
        # Misc
        "clear_cache_interval": clear_cache_interval,
    }

    hparams.add_params(prefix="data", **data_config)
    hparams.add_params(prefix="model", **model_config)
    hparams.add_params(prefix="optim", **optimizer_config)


    assert n_steps >= hparams.optim.lr_warmup_steps + hparams.optim.lr_decay_steps, f"Number of steps {n_steps} must be greater than warmup steps {hparams.optim.lr_warmup_steps} + decay steps {hparams.optim.lr_decay_steps}"
    if hparams.optim.lr_min_scale == 0.0:
        assert n_steps == hparams.optim.lr_warmup_steps + hparams.optim.lr_decay_steps, f"Learning rate goes to zero before training finishes. Set lr_min_scale to a value greater than 0.0 or increase decay steps"

    if debug:
        hparams.run = f"debug_{hparams.run}"

    trainer = ArcTrainer(hparams=hparams,
                        parent_dir=_BASE_DIR,
                        prevent_overwrite=True,
                        disable_checkpointing_and_logging=True if (lr_find or debug) else False)
    
    if checkpoint is not None:
        existing_checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)
        assert existing_checkpoint is None, f"Checkpoint {existing_checkpoint} already exists. Loading from checkpoint will overwrite the existing checkpoint"
        trainer.initialise_from_checkpoint(checkpoint, 
                                           strict=False, 
                                           load_model=True, 
                                           load_step=False,     # NO RESUME, start from the beginning 
                                           load_optim=False)    # NO RESUME, start from the beginning 

    if lr_find:
        trainer.find_lr()
    else:
        trainer.train(max_steps=n_steps)


@app.command("fork")
def fork(
        experiment: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        checkpoint: str = typer.Argument(..., help="Path to the checkpoint file to fork training from"),

        # Batch Config
        tbs: Optional[int] = typer.Option(None, min=1, help="Train Batch Size (in tokens)"),
        ebs: Optional[int] = typer.Option(None, min=1, help="Eval Batch Size (in tokens)"),
        grad_accum: Optional[int] = typer.Option(None, min=1, help="Number of steps to accumulate gradients over."),

        # Model Config
        n_iter: Optional[int] = typer.Option(None, min=2, help="Number of iterations for the model"),

        # Loss Config
        gamma: Optional[float] = typer.Option(None, min=1.0, help="Loss Gamma. Gamma == 1 is equivalent to Cross Entropy Loss"),
        lalpha: Optional[float] = typer.Option(None, min=0.0, max=0.5, help="Portion of inverse loss, 0 <= lalpha <= 0.5"),
        # edr: Optional[float] = typer.Option(None, min=0.0, help="Loss Error Decay Rate"),
        # mctp: Optional[float] = typer.Option(None, min=0.0, help="Min Correct Tokens Percentage"),

        # Misc Config
        n_steps: Optional[int] = typer.Option(1_000_000, min=1, help="Number of steps to train for. If None, lr_decay + lr_warmup is used"),
        seed: Optional[int] = typer.Option(None, min=0, help="Random seed for the data and experiment"),

        # Learning Rate Config (If any of the Learning Rate Config is changed, then resume must be false)
        mlr: Optional[float] = typer.Option(None, min=0.0, help="Learning Rate"),
        plr: Optional[float] = typer.Option(None, min=0.0, help="Program Learning Rate. If None, then it is set to mlr"),
        lr_warmup: Optional[int] = typer.Option(None, min=0, help="Number of steps for learning rate warmup. Only used for noam and lindecay scheduler"),
        lr_min_scale: Optional[float] = typer.Option(None, min=0.0, help="Learning Rate reached after decay phase is obtained by scaling the max learning rate by this factor"),
        lr_decay: Optional[int] = typer.Option(None, min=0, help="Number of steps for learning rate decay. Only used for noam and lindecay scheduler"),
        lr_schedule: Optional[LRSchedule] = typer.Option(None, help="Learning rate scheduler. Options: noam, alt, const"),
        plt_patience: Optional[int] = typer.Option(None, min=0, help="Patience for plateau scheduler to reduce learning rate"),
        plt_factor: Optional[float] = typer.Option(None, min=0.0, help="Factor for plateau scheduler to reduce learning rate"),
        plt_warmup: Optional[int] = typer.Option(None, min=0, help="Warm up in number of updates before Pleateau Scheduler starts tracking Plateau metric"),
        resume: Optional[bool] = typer.Option(True, help="Resume training from the checkpoint. If False, then optimizer and scheduler are not loaded, only model is loaded"),

        # Regularisation/ Weight Decay Config
        mwd: Optional[float] = typer.Option(None, min=0.0, help="Weight Decay"),
        pwd: Optional[float] = typer.Option(None, min=0.0, help="Program Weight Decay"),
        pl1: Optional[float] = typer.Option(None, min=0.0, help="Program L1 Regularization"),
        dropout: Optional[float] = typer.Option(None, min=0.0, max=1.0, help="Dropout probability"),

        # Grok Config
        grok_alpha: Optional[float] = typer.Option(None, min=0.0, help="Grok Alpha"),
        grok_lambda: Optional[float] = typer.Option(None, min=0.0, help="Grok Lambda"),

        # Data Config
        min_train_pp: Optional[int] = typer.Option(None, help="Minimum number of Train Examples Per Program"),
        max_train_pp: Optional[int] = typer.Option(None, help="Maximum number of Train Examples Per Program"),
        min_test_pp: Optional[int] = typer.Option(None, help="Minimum number of Test Examples Per Program"),
        max_test_pp: Optional[int] = typer.Option(None, help="Maximum number of Test Examples Per Program"),
        include_train: Optional[bool] = typer.Option(None, help="Include training data for training"),
        include_aux: Optional[bool] = typer.Option(None, help="Include auxiliary data for training"),
        include_eval: Optional[bool] = typer.Option(None, help="Include evaluation data for training"),
        include_inv: Optional[bool] = typer.Option(None, help="Include inverse data for training"),
        permute: Optional[bool] = typer.Option(None, help="Permute the training set for each batch"),

        # Misc Config
        eval_int: Optional[int] = typer.Option(None, help="Number of steps between evaluations. None means evaluation at the end of each epoch"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        debug: Optional[bool] = typer.Option(False, help="For test runs. Nothing is saved"),
    ):

    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    hparams = ArcHparams.from_dict(hparams_dict, build_state=False)

    prev_exp = f"{hparams.experiment}_{hparams.run}"
    new_exp = f"{experiment}_{run}"

    is_resume = prev_exp == new_exp
    if is_resume:
        assert resume, "If resuming from the same run, then resume must be true"

    base_config = {
        "experiment": experiment,
        "run": run,
        "seed": seed,
        "grok_alpha": grok_alpha,
        "grok_lambda": grok_lambda,
        "eval_interval": eval_int,
        "target_metric": 'ARC_EVAL/Accuracy(%)' if include_eval else 'ARC_TRAIN/Accuracy(%)',
        "plateau_patience": plt_patience,
        "plateau_factor": plt_factor,
        "plt_warmup": plt_warmup,
        "accumulation_steps": grad_accum,

    }


    data_config = {
        'include_train': include_train,
        'include_aux': include_aux,
        'include_eval': include_eval,
        'include_inv': include_inv,
        'min_train_pp': min_train_pp,
        'max_train_pp': max_train_pp,
        'min_test_pp': min_test_pp,
        'max_test_pp': max_test_pp,
        'permute': permute
    }

    model_config = {
        "n_iter": n_iter,
        "gamma": gamma,
        "lalpha": lalpha
    }

    if any(option is not None for option in [mlr, plr, lr_decay, lr_min_scale, lr_warmup, lr_schedule, plt_patience, plt_factor]):
        assert not resume, "If any of the Learning Rate Config is changed, then resume must be false. And no optimizer and scheduler are loaded, only model is loaded"


    optimizer_config = {
        # Batch Size
        "train_batch_token_count": tbs,
        "eval_batch_token_count": ebs,

        # Regularization / Weight Decay
        "wd_model": mwd,
        "wd_prog": pwd,
        "dropout": dropout,
        "l1_prog": pl1,

        # Compute / Loss Config
        # "edr": edr,
        # "mctp": mctp,

        # Learning Rate
        "lr_model": mlr if not lr_find else (1 if mlr > 0 else 0),
        "lr_prog": plr if not lr_find else (1 if plr > 0 else 0),

        "lr_schedule": lr_schedule.value if isinstance(lr_schedule, LRSchedule) else None,
        "lr_warmup_steps": lr_warmup,
        "lr_min_scale": lr_min_scale,
        "lr_decay_steps": lr_decay
    }

    def override_hparams(hparams, config):
        for key, value in config.items():
            if value is not None:
                setattr(hparams, key, value)


    override_hparams(hparams, base_config)
    override_hparams(hparams.data, data_config)
    override_hparams(hparams.model, model_config)
    override_hparams(hparams.optim, optimizer_config)

    assert hparams.data.include_eval or hparams.data.include_train, "At least one of include_eval or include_train must be True"


    assert n_steps >= hparams.optim.lr_warmup_steps + hparams.optim.lr_decay_steps, f"Number of steps {n_steps} must be greater than warmup steps {hparams.optim.lr_warmup_steps} + decay steps {hparams.optim.lr_decay_steps}"
    if hparams.optim.lr_min_scale == 0.0:
        assert n_steps == hparams.optim.lr_warmup_steps + hparams.optim.lr_decay_steps, f"Learning rate goes to zero before training finishes. Set lr_min_scale to a value greater than 0.0 or increase decay steps"


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
        trainer.info(f"Attemping to resume {experiment}/{run}")

    trainer.initialise_from_checkpoint(checkpoint,
                                    strict=False, 
                                    load_model=True, 
                                    load_step=True,     # When forking, we always load the step, irrespective of resume
                                    load_optim=resume)  # If resume is False, then only model is loaded, not optimizer and scheduler 


    if lr_find:
        trainer.find_lr()
    else:
        trainer.train(max_steps=n_steps)


@app.command("info")
def info(
       run_path: str = typer.Argument(..., help="Path to the run folder (not checkpoint) to resume training from"),
    ):
    experiment, run, parent_dir = split_run_path(run_path)
    checkpoint_dir = ArcTrainer.get_checkpoint_dir(experiment, run, parent_dir=parent_dir)
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
    checkpoint = ArcTrainer.get_latest_checkpoint(checkpoint_dir)       
    hparams_dict = ArcTrainer.load_hparams_dict(checkpoint)
    print(hparams_dict)


if __name__ == "__main__":
    app()
