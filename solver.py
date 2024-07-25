from typing import Optional, Tuple

import typer


from src.arc_trainer import ArcTrainer, ArcHparams
from rich import print

app = typer.Typer(pretty_exceptions_show_locals=False)
train_app = typer.Typer()
app.add_typer(train_app, name="train")





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
    checkpoint = trainer.get_latest_checkpoint(trainer.checkpoint_dir)
    assert checkpoint is None, f"Checkpoint {checkpoint} already exists. Use the train resume command to resume training"
    
    if lr_find:
        trainer.find_lr()
    else:
        trainer.train()


@train_app.command("resume")
def resume(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
    ):
    
    checkpoint_dir = ArcTrainer.get_checkpoint_dir(name, run)
    assert checkpoint_dir.exists(), f"Checkpoint directory {checkpoint_dir} does not exist"
    checkpoint = ArcTrainer.get_latest_checkpoint(checkpoint_dir)       
    trainer = ArcTrainer.from_checkpoint(checkpoint, resume=False)
    trainer.train()





if __name__ == "__main__":
    app()
