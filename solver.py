import json
from typing import Optional
import typer
import logging

from src.dataset import TrainingData
from src.interpreter import Interpreter, InterpreterConfig
from src.arc_trainer import ArcTrainer
from src.utils import nearest_greater_power_of_2
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")



def create_training_data(data_aug: Optional[int] = 3, sep_task_version: bool = True, seed: int = 42):


    print("--"*20)
    print("TRAINING DATA CONFIGURATION")
    print("--"*20)
    print(f"Data Augmentation Level: {data_aug}")
    print(f"Separate Task Version: {sep_task_version}")
    print(f"Seed: {seed}")
    training_data = TrainingData(augmentation_factor=data_aug,
                                join_version=not sep_task_version, 
                                seed=seed).load()
    
    print(f"Number of Training Examples: {len(training_data.train_ds)}")
    print(f"Number of Evaluation Examples: {len(training_data.eval_ds)}")
    print("--"*20)
    return training_data
    

def create_model(prog_tokenizer, grid_tokenizer, n_dim, n_heads, n_mixers, n_blocks, n_layers, share_mixer) -> Interpreter:
    prog_vocab_size = nearest_greater_power_of_2(len(prog_tokenizer))
    grid_vocab_size = nearest_greater_power_of_2(len(grid_tokenizer))

    config = InterpreterConfig(
        prog_vocab_size = prog_vocab_size,
        grid_vocab_size = grid_vocab_size,
        n_dim = n_dim, # dimension of the model
        n_head = n_heads, # number of heads within each self-attention block
        n_mixers = n_mixers, # number of self-attention layers within each transformer block
        n_blocks = n_blocks, # number of transformer blocks within each recurrence block
        n_rec_layers = n_layers, # number of recurrences
        share_mixer=share_mixer
    )
    print("--"*20)
    print("MODEL CONFIGURATION")
    print("--"*20)
    print(f"{json.dumps(config.to_dict(), indent=4)}")


    model = Interpreter(config,
                    prog_tokenizer=prog_tokenizer,
                    grid_tokenizer=grid_tokenizer)
    print("--"*20)
    return model


@train_app.command("new")
def train(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
        mlr: float = typer.Argument(0.01, min=-1.0, help="Learning Rate. If -1, then learning rate finder is invoked in debug model."),
        mwd: float = typer.Argument(0.01, min=0.0, help="Weight Decay"),
        pwd: float = typer.Argument(0.0, min=0.0, help="Program Weight Decay"),
        n_dim: int = typer.Argument(128, min=8, max=512, help="Dimension of the model"),
        n_heads: int = typer.Argument(16, min=1, max=64, help="Number of heads within each self-attention block"),
        n_blocks: int = typer.Argument(3, min=1, max=20, help="Number of mixing blocks within each recurrent layer"),
        n_mixers: int = typer.Argument(3, min=1, max=10, help="Number of mixers within each mixing block"),
        n_layers: int = typer.Argument(3, min=1, max=10, help="Number of recurrent layers"),
        plr: Optional[float] = typer.Argument(None, min=-1.0, help="Program Learning Rate. If None, then it is scaled according to data augmentation level"),
        data_aug: int = typer.Option(3, min=0, help="Data Augmentation Level. 0 means no augmentation"),
        debug: bool = typer.Option(False, min=0, help="For test runs. Nothing saved."),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment"),
        sep_task_version: bool = typer.Option(True, help="If set, task ID and task version are given separate embeddings"),
        share_mixer: bool = typer.Option(True, help="Share mixer within each mixing block"),
        lr_find: bool = typer.Option(False, help="Run learning rate finder in debug mode"),
        device: Optional[str] = typer.Option(None, help="Device to run the training on. If None, then it is automatically selected"),
    ):

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, ) as progress:
        progress.add_task(description="Loading Training Data ...", total=None)
        training_data = create_training_data(data_aug, sep_task_version, seed)
        
        program_tokenizer = training_data.program_tokenizer
        grid_tokenizer = training_data.grid_tokenizer

        progress.add_task(description="Initialising Model ...", total=None)
        model = create_model(prog_tokenizer=program_tokenizer,
                            grid_tokenizer=grid_tokenizer,
                            n_dim=n_dim,
                            n_heads=n_heads,
                            n_blocks=n_blocks,
                            n_mixers=n_mixers,
                            n_layers=n_layers,
                            share_mixer=share_mixer)
        
        if plr is None:
            # There is factor of 8 samples for every level of data augmentation
            plr_scale = 1 if data_aug <= 0 else 8 * data_aug
            plr = mlr * plr_scale
            print(f"Setting Program Learning Rate: {plr}")

        optimizer = model.get_optimizer(model_lr=mlr,
                                        model_wd=mwd,
                                        prog_lr=plr,
                                        prog_wd=pwd,
                                        device_type=device)
        


        # trainer = ArcTrainer(
        #     experiment_name=name,
        #     run_name=run,
        #     eval_interval=10,
        #     num_epochs=1000,
        #     model=model,
        #     hparams=config,
        #     optimizer=optimizer,
        #     train_dl=train_dl,
        #     eval_dl=eval_dl,
        #     log_level=logging.INFO,
        #     disable_checkpointing_and_logging=debug
        #     )
        
        # if lr_find:
        #     trainer.find_lr()
        # else:
        #     trainer.train()


@train_app.command("resume")
def resume(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
    ):
    pass

@train_app.command("from")
def from_checkpoint(
        name: str = typer.Argument(..., help="Name of the experiment saved at `./runs/{name}`"),
        run: str = typer.Argument(..., help="Name of the run within the experiment saved at `./runs/{name}/{run}`"),
    ):
    pass



if __name__ == "__main__":
    app()
