import json
from typing import Optional
import typer

from src.dataset import TrainingData
from src.interpreter import Interpreter, InterpreterConfig
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
    print("--"*20)
    return training_data
    

def create_model(prog_tokenizer, grid_tokenizer, n_dim, n_heads, n_mixers, n_blocks, n_layers, share_mixer):
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
        n_dim: int = typer.Argument(128, min=8, max=512, help="Dimension of the model"),
        n_heads: int = typer.Argument(16, min=1, max=64, help="Number of heads within each self-attention block"),
        n_blocks: int = typer.Argument(3, min=1, max=20, help="Number of mixing blocks within each recurrent layer"),
        n_mixers: int = typer.Argument(3, min=1, max=10, help="Number of mixers within each mixing block"),
        n_layers: int = typer.Argument(3, min=1, max=10, help="Number of recurrent layers"),
        debug: bool = typer.Option(False, min=0, help="For test runs. Nothing saved."),
        data_aug: Optional[int] = typer.Option(3, min=0, help="Data Augmentation Level"),
        seed: int = typer.Option(42, min=0, help="Random seed for the data and experiment"),
        sep_task_version: bool = True,
        share_mixer: bool = typer.Option(True, help="Share mixer within each mixing block"),
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


@train_app.command("resume")
def resume(
        name: str = typer.Argument(..., help="Name of the experiment. Added to runs/"),
        run: str = typer.Argument(..., help="Name of the run within the experiment. Created as subdirectory within experiment directory"),
    ):
    pass

@train_app.command("from")
def from_checkpoint(
        name: str = typer.Argument(..., help="Name of the experiment. Added to runs/"),
        run: str = typer.Argument(..., help="Name of the run within the experiment. Created as subdirectory within experiment directory"),
    ):
    pass



if __name__ == "__main__":
    app()
