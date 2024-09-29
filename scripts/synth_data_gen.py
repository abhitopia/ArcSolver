
#%%
import json
from pathlib import Path
from typing import List, Union
import numpy as np

from src.task import ArcTask
from src.task import ArrayTransform, ColorPermutation
#%%


def generate_synth_scale_data(num_tasks=400, dest_path: Path = 'data/synthetic/'):
    def determine_scale_factors():
        scale_x = np.random.choice([1, 2, 3, 4, 5])
        scale_y = np.random.choice([1, 2, 3, 4, 5])
        if scale_x == 1 and scale_y == 1:
            return determine_scale_factors()
        return scale_x, scale_y

    def generate_random_grid(max_dim, scale_x, scale_y):
        max_width = max_dim // scale_x
        max_height = max_dim // scale_y
        width = np.random.randint(1, max_width)
        height = np.random.randint(1, max_height)
        grid = np.random.randint(0, 10, (height, width))
        return grid

    def scale_grid(input_grid, scale_x, scale_y):
        output_grid = np.repeat(np.repeat(input_grid, scale_y, axis=0), scale_x, axis=1)
        return output_grid
    
    def gen_task(num_train=3, num_test=1, scale_down=False):
        # Generate the input and output grids
        scale_x, scale_y = determine_scale_factors()

        def fnc(num_examples, scale_x, scale_y):
            examples = []
            for _ in range(num_examples):
                input_grid = generate_random_grid(max_dim=30,
                                                scale_x=scale_x, 
                                                scale_y=scale_y)

                output_grid = scale_grid(input_grid, scale_x, scale_y)

                if scale_down:
                    input_grid, output_grid = output_grid, input_grid

                examples.append({'input': input_grid, 'output': output_grid})
            return examples
    
        train, test = fnc(num_train, scale_x, scale_y), fnc(num_test, scale_x, scale_y)
        return train, test, scale_x, scale_y
    
    def generate_synthetic_dataset(num_tasks=400, scale_down=False, name=None):
        tasks = []
        for i in range(num_tasks):
            train, test, scale_x, scale_y = gen_task(scale_down=scale_down)
            task = ArcTask(i, train, test, dataset=f"{name}_{scale_x}_{scale_y}")
            tasks.append(task)
        
        return tasks

    # base_path = Path(__file__).resolve().parent.parent    
    # dest_path = base_path / Path(dest_path)
    dest_path = Path(dest_path)
    print(dest_path)
    num_tasks = 400
    for scale_down in [False, True]:
        if scale_down:
            dataset_name = "SCLDN"
        else:
            dataset_name = "SCLUP"

        dataset_path = dest_path
        tasks = generate_synthetic_dataset(num_tasks=num_tasks, scale_down=scale_down, name=dataset_name)

        for task in tasks:
            task_dict = task.to_dict()
            task_path = dataset_path / task.dataset /f"{task.id}.json"
            task_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(task_dict, task_path.open('w'), indent=4)


#%%
generate_synth_scale_data(dest_path='/Users/abhishekaggarwal/synced_repos/ArcSolver/data/synthetic/')
# %%
def gen_synth_arc_data(num_tasks=400, dest_path: Path = 'data/synthetic/'):

    def gen_grid(max_size=30):
        # Define the dimensions
        width = np.random.randint(2, max_size+1)  # Randomly choose width between 2 and 30
        height = np.random.randint(2, max_size+1)  # Randomly choose height between 2 and 30

        # Generate the 2D grid with random values between 0 and 9
        grid = np.random.randint(0, 10, size=(height, width))
        return grid

    def gen_task(transformation, num_train=4, num_test=1):
        # Generate the input and output grids

        max_size = np.random.randint(2, 31)
        def fnc(num_examples):
            examples = []
            for _ in range(num_examples):
                input_grid = gen_grid(max_size=max_size)
                output_grid = transformation(input_grid)
                examples.append({'input': input_grid, 'output': output_grid})
            return examples
    
        train, test = fnc(num_train), fnc(num_test)
        return train, test
    

    def generate_synthetic_dataset(transform: Union[callable, List[callable]], num_tasks=400, name=None):
        if not isinstance(transform, list):
            transform = [transform]
        
        tasks = []
        for i in range(num_tasks):
            transform_id = i % len(transform)
            task_transform = transform[transform_id]
            train, test = gen_task(task_transform)

            dataset = f"{name}_{transform_id}" if len(transform) > 1 else name
            task = ArcTask(i, train, test, dataset=dataset)
            tasks.append(task)
        
        return tasks

    # base_path = Path(__file__).resolve().parent.parent    
    dest_path = Path(dest_path)
    num_tasks = 400
    for transformation in list(ArrayTransform) + [list(ColorPermutation)]:
        if isinstance(transformation, list):
            transforms = [t.transform for t in transformation]
            dataset_name = f"CLRPM"
        else:
            transforms = [transformation.transform]
            dataset_name = f"{transformation.name}"

        dataset_path = dest_path
        tasks = generate_synthetic_dataset(transforms, num_tasks=num_tasks, name=dataset_name)

        for task in tasks:
            task_dict = task.to_dict()
            task_path = dataset_path / task.dataset / f"{task.id}.json"
            task_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(task_dict, task_path.open('w'), indent=4)


gen_synth_arc_data(dest_path='/Users/abhishekaggarwal/synced_repos/ArcSolver/data/synthetic/')
#%%
