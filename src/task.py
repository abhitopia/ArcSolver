
#%%
from enum import Enum, auto
import json
from pathlib import Path
from typing import List, Optional
import numpy as np


class ArcTask:
    def __init__(self, id, train, test, dataset=None, version=None, augmentation_group="None"):
        self.dataset = dataset
        self.version = version
        self.augmentation_group = augmentation_group
        self.id = id
        self.train = [(np.array(example['input']), np.array(example['output'])) for example in train]
        self.test = [(np.array(example['input']), np.array(example['output'])) for example in test]

    def __repr__(self):
        return f'ArcTask(id={self.id}, dataset={self.dataset}, version={self.version}, augmentation_group={self.augmentation_group})'

    def __eq__(self, other):
        if not isinstance(other, ArcTask):
            return False

        def check_examples(examples1, examples2):
            if len(examples1) != len(examples2):
                return False

            for i in range(len(examples1)):
                if not np.array_equal(examples1[i][0], examples2[i][0]):
                    return False
                if not np.array_equal(examples1[i][1], examples2[i][1]):
                    return False
            return True
        
        return check_examples(self.train, other.train) and check_examples(self.test, other.test)
    
    def __lt__(self, other):
        if not isinstance(other, ArcTask):
            return NotImplemented
        return (self.id, self.version) < (other.id, other.version)



class ColorPermutation(Enum):
    CPID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Identity
    CP01 = [7, 4, 5, 2, 8, 3, 0, 9, 6, 1]
    CP02 = [0, 9, 4, 5, 6, 8, 1, 3, 2, 7]
    CP03 = [7, 4, 1, 9, 6, 0, 8, 2, 5, 3]
    CP04 = [9, 6, 5, 7, 4, 0, 3, 8, 1, 2]
    CP05 = [1, 8, 0, 3, 9, 5, 6, 2, 7, 4]
    CP06 = [5, 3, 1, 9, 7, 6, 0, 2, 8, 4]
    CP07 = [1, 4, 3, 8, 7, 9, 6, 2, 5, 0]
    CP08 = [6, 0, 2, 1, 3, 4, 7, 8, 5, 9]
    CP09 = [2, 0, 3, 8, 4, 6, 1, 9, 5, 7]

    @property
    def transform(self):
        colors = self.value
        color_mapping = {original: new for original, new in enumerate(colors)}
        return lambda x: np.vectorize(color_mapping.get)(x)


class ArrayTransform(Enum):
    IDENT = auto()
    RT090 = auto()
    RT180 = auto()
    RT270 = auto()
    FLPLR = auto()
    FLPUD = auto()
    FLPDG = auto()
    FLPAD = auto()

    @property
    def transform(self):
        return {
            'IDENT': lambda x: x,
            'RT090': lambda x: np.rot90(x),
            'RT180': lambda x: np.rot90(x, k=2),
            'RT270' : lambda x: np.rot90(x, k=3),
            'FLPLR': lambda x: np.fliplr(x),
            'FLPUD': lambda x: np.flipud(x),
            'FLPDG': lambda x: np.flipud(np.rot90(x)),
            'FLPAD': lambda x: np.fliplr(np.rot90(x)),           
        }[self.name]


class TaskInvariantTransform:
    def __init__(self, array_transform: ArrayTransform, color_perm: ColorPermutation):
        self.arr_transform = array_transform
        self.color_perm = color_perm
    
    @property
    def name(self):
        return f'{self.arr_transform.name}_{self.color_perm.name}'

    def __call__(self, task: ArcTask) -> ArcTask:

        task_copy = ArcTask(id=task.id,
                            train=[],
                            test=[],
                            dataset=task.dataset,
                            version=self.name)
        
        task_copy.train = task.train
        task_copy.test = task.test

        for transform in [self.arr_transform.transform, self.color_perm.transform]:
            task_copy.train = [(transform(example[0]), transform(example[1])) for example in task_copy.train]
            task_copy.test = [(transform(example[0]), transform(example[1])) for example in task_copy.test]

        return task_copy
    

class TaskAugmenter:
    def __init__(self, augmentation_id: int):
        self.transformation_groups = TaskAugmenter.transformation_groups()
        assert 0 <= augmentation_id < len(self.transformation_groups), f'Invalid augmentation_id: {augmentation_id}'
        self.group_id = augmentation_id
        
    @staticmethod
    def transformation_groups():
        List_A = list(ArrayTransform)
        List_B = list(ColorPermutation)
        groups = []
        for i in range(len(List_B)):
            group = [(List_A[j % len(List_A)], List_B[(j + i) % len(List_B)]) for j in range(len(List_A))]
            groups.append(group)

        return groups

    def __call__(self, task: ArcTask, remove_redundant=True) -> ArcTask:
        group = self.transformation_groups[self.group_id]
        augmented_tasks = []
        array_transformed_tasks = []
        for array_transform, color_transform in group:
            # Apply only the array transformation
            no_color_transform = TaskInvariantTransform(array_transform, ColorPermutation.CPID)
            array_transformed_tasks.append(no_color_transform(task))
            transform = TaskInvariantTransform(array_transform, color_transform)
            transformed_task = transform(task)

            # Change the dataset name to include the transformation group id
            transformed_task.augmentation_group = f'{self.group_id:02d}'
            augmented_tasks.append(transformed_task)

        if remove_redundant:
            # Remove all redundant tasks where array transformation is the same
            redundant_tasks = set()
            for i in range(len(array_transformed_tasks)):
                if i in redundant_tasks:
                    continue
                for j in range(i+1, len(array_transformed_tasks)):
                    if array_transformed_tasks[i] == array_transformed_tasks[j]:
                        redundant_tasks.add(j)

            augmented_tasks = [task for i, task in enumerate(augmented_tasks) if i not in redundant_tasks]

        return augmented_tasks
    

class ArcTasksLoader:
    def __init__(self, name: str, path: str):
        base_path = Path(__file__).resolve().parent.parent    
        self.path = base_path / Path(path)
        assert self.path.exists(), f'Path does not exist: {self.path}'
        self.name = name

    def __lt__(self, other):
        if not isinstance(other, ArcTasksLoader):
            return NotImplemented
        return self.name < other.name

    def json_files(self):
        return [json for json in Path(self.path).glob("**/*.json")]
    
    def _load_tasks(self) -> List[ArcTask]:
        tasks = []
        for f in self.json_files():
            task_json = json.load(f.open('r'))
            task_id = self.name + "_" + f.stem
            task = ArcTask(id=task_id,
                           train=task_json['train'],
                           test=task_json['test'],
                           dataset=self.name,
                           version='original')
            tasks.append(task)
        return tasks
    
    def load_tasks(self, augmentation_id: Optional[int] = None) -> List[ArcTask]:
        tasks = self._load_tasks()

        if augmentation_id is None:
            return tasks
        
        assert isinstance(augmentation_id, int), 'augmentation_id must be an integer'
        assert 0 <= augmentation_id < len(TaskAugmenter.transformation_groups()), f'Invalid augmentation_id: {augmentation_id}'
        
        augmented_tasks = []
        task_augmenter = TaskAugmenter(augmentation_id)
        for task in tasks:
            augmented_tasks.extend(task_augmenter(task))

        # Returns after sorting so results are reproducible
        return sorted(augmented_tasks)


# Training Tasks
ARC_1D = ArcTasksLoader(name='1D-ARC', path='data/arc_dataset_collection/dataset/1D-ARC/data')
ARC_TRAIN = ArcTasksLoader(name='ARC_TRAIN', path='data/arc_dataset_collection/dataset/ARC/data/training')
ARC_SYTH_EXTEND = ArcTasksLoader(name='ARC_SYTH_EXTEND', path='data/arc_dataset_collection/dataset/ARC_synthetic_extend/data')
ARC_COMMUNITY = ArcTasksLoader(name='ARC_COMMUNITY', path='data/arc_dataset_collection/dataset/arc-community/data')
ARC_DIVA = ArcTasksLoader(name='ARC_DIVA', path='data/arc_dataset_collection/dataset/arc-dataset-diva/data')
ARC_CONCEPT = ArcTasksLoader(name='ARC_CONCEPT', path='data/arc_dataset_collection/dataset/ConceptARC/data')
ARC_DBIGHAM = ArcTasksLoader(name='ARC_DBIGHAM', path='data/arc_dataset_collection/dataset/dbigham/data')
ARC_MINI = ArcTasksLoader(name='ARC_MINI', path='data/arc_dataset_collection/dataset/Mini-ARC/data')
ARC_NOSOUND = ArcTasksLoader(name='ARC_NOSOUND', path='data/arc_dataset_collection/dataset/nosound/data')
ARC_PQA = ArcTasksLoader(name='ARC_PQA', path='data/arc_dataset_collection/dataset/PQA/data')
ARC_REARC_EASY = ArcTasksLoader(name='ARC_REARC_EASY', path='data/arc_dataset_collection/dataset/RE-ARC/data/easy')
ARC_REARC_HARD = ArcTasksLoader(name='ARC_REARC_HARD', path='data/arc_dataset_collection/dataset/RE-ARC/data/hard')
ARC_SEQUENCE = ArcTasksLoader(name='ARC_SEQUENCE', path='data/arc_dataset_collection/dataset/Sequence_ARC/data')
ARC_SYNTH_RIDDLES = ArcTasksLoader(name='ARC_SYNTH_RIDDLES', path='data/arc_dataset_collection/dataset/synth_riddles/data')
ARC_EVAL = ArcTasksLoader(name='ARC_EVAL', path='data/arc_dataset_collection/dataset/ARC/data/evaluation')


TRAINING_TASKLOADER = ARC_TRAIN
EVALUATION_TASKLOADER = ARC_EVAL

AUXILIARY_TASKLOADERS = [
    ARC_1D,
    ARC_SYTH_EXTEND,
    ARC_COMMUNITY,
    ARC_DIVA,
    ARC_CONCEPT,
    ARC_DBIGHAM,
    ARC_MINI,
    ARC_NOSOUND,
    ARC_PQA,
    ARC_REARC_EASY,
    ARC_REARC_HARD,
    ARC_SEQUENCE,
    ARC_SYNTH_RIDDLES
]


# Evaluation Tasks
#%%

# %%
