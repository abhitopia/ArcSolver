#%%
import json
import pickle
import random
from pathlib import Path
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, BatchSampler, DataLoader
import torch
import numpy as np
from .utils import hash_string
from .task import TRAINING_TASKLOADER, ArcTasksLoader, AUXILIARY_TASKLOADERS, ArcTask

BASE_PATH = Path(__file__).resolve().parent.parent
CACHE_FOLDER = BASE_PATH / '.cache'


class Tokenizer:
    def __init__(self, token2idx={}, idx2token={}, frozen=True) -> None:
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.frozen = frozen
    
    def add_token(self, token):
        if self.frozen:
            raise ValueError('Tokenizer is frozen. No new tokens can be added.')
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    def encode(self, sequence: str):
        sequence = sequence.split(' ')
        return [self.token2idx[token] for token in sequence]

    def decode(self, sequence, remove_padding=True):
        tokens = [self.idx2token[idx] for idx in sequence]
        return ' '.join(tokens)
    
    def to_dict(self):
        return {
            'token2idx': self.token2idx,
            'idx2token': self.idx2token,
            'frozen': self.frozen
        }

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.token2idx = data['token2idx']
        obj.idx2token = data['idx2token']
        obj.frozen = data['frozen']
        return obj
    
    def __eq__(self, value: object) -> bool:
        assert isinstance(value, Tokenizer), 'value must be an instance of Tokenizer'
        return self.token2idx == value.token2idx and self.idx2token == value.idx2token

    def __len__(self):
        return len(self.token2idx)
    

class GridTokenizer(Tokenizer):
    def __init__(self):
        self.PAD_TOKEN = '<NOOP>'
        tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[[', '],[', ']]', self.PAD_TOKEN]
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        super().__init__(token2idx=token2idx, idx2token=idx2token, frozen=True)
        self.PAD_IDX = self.token2idx[self.PAD_TOKEN]

    def decode(self, sequence, remove_padding=True):
        tokens = super().decode(sequence)
        if remove_padding:
            tokens = [token for token in tokens.split(' ') if token != self.PAD]
        return ' '.join(tokens)


class ProgramTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(frozen=False)

    def build(self, tokens: List[str]):
        if self.frozen:
            raise ValueError('Tokenizer is frozen. No new tokens can be added.')
        for token in tokens:
            for t in token.strip().split(' '):
                self.add_token(t)
        self.frozen = True


class TaskToExamples:
    def __init__(self, join_version: bool = True):
        self.join_version = join_version

    @staticmethod
    def serialize_array(array: np.ndarray) -> str:
        list_of_lists = array.tolist()
        array_str = json.dumps(list_of_lists)
        array_str = array_str.replace('\n', '').replace(',','')
        return array_str.replace('[[', '[[ ').replace(']]', ' ]]').replace('] [', ' ],[ ')

    @staticmethod
    def deserialize_array(array_str: str) -> np.ndarray:
        rows = array_str.split('],[')
        rows[0] = rows[0][3:]
        rows[-1] = rows[-1][:-3]
        
        result = []
        for row in rows:
            result.append('[' + ', '.join(row.strip().split(' ')) + ']')
        result = "[" + ",".join(result) + "]"
        result = json.loads(result)
        return np.array(result)

    def __call__(self, task: ArcTask) -> Tuple[List[str], List[str]]:
        program = [task.id, task.version]
        program = '_'.join(program) if self.join_version else ' '.join(program)

        def get_examples(inp_out_pairs):
            examples = []
            for inp, out in inp_out_pairs:
                inp_str = self.serialize_array(inp)
                out_str = self.serialize_array(out)
                example = ((program, inp_str), out_str)
                examples.append(example)
            return examples

        return get_examples(task.train), get_examples(task.test)


class TargetTokenCountBatchSampler(BatchSampler):
    def __init__(self, dataset: Dataset, target_token_count: int, shuffle: bool = True):
        self.dataset = dataset
        self.target_token_count = target_token_count
        self.shuffle = shuffle
        self.batches = self.create_batches()

    def shuffle_batches(self):
        # Shuffle the order of the data within each batch
        for batch in self.batches:
            random.shuffle(batch)
        # Shuffle the order of the batches
        random.shuffle(self.batches)

    @staticmethod
    def compute_example_len(example):
        (p, i), o = example
        return len(p) + max(len(i), len(o))

    def create_batches(self):
        sorted_indices = sorted(range(len(self.dataset)), key=lambda i: self.compute_example_len(self.dataset[i]))
        batches = [] 
        batch = []
        max_batch_len = 0
        target_token_count = self.target_token_count

        for idx in sorted_indices:
            example_len = self.compute_example_len(self.dataset[idx])
            include_token_count = max(example_len, max_batch_len) * (len(batch) + 1)
            exclude_token_count = max_batch_len * len(batch)

            if idx == len(sorted_indices) - 1:
                batch.append(idx)
                batches.append(batch)
            elif abs(include_token_count - target_token_count) > abs(exclude_token_count - target_token_count):
            # elif include_token_count > target_token_count:
                batches.append(batch)
                batch = [idx]
                max_batch_len = example_len
            else:
                batch.append(idx)
                max_batch_len = max(max_batch_len, example_len)

        return batches

    
    def __iter__(self):
        # Shuffle the buckets and data within buckets if necessary
        if self.shuffle:
            self.shuffle_batches()

        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)



class ArcExamplesDataset(Dataset):
    def __init__(self, examples: List[Tuple[Tuple[List[int], List[int]], List[int]]], pad_idx: int):
        self.examples = examples
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def subset(self, num_examples: int):
        """
        Returns a subset of the dataset
        """
        if num_examples <= 0:
            return self
        
        indices = list(range(num_examples))
        return ArcExamplesDataset([self.examples[i] for i in indices], pad_idx=self.pad_idx)
    
    @property
    def max_example_seq_len(self):
        return max([len(p) + max(len(i), len(o)) for ((p, i), o) in self.examples])

    
    @staticmethod
    def collate_fn(batch, pad_idx, seq_length: Optional[int] = None, device=torch.device('cpu')):
        programs_inputs, outputs = zip(*batch)
        programs, inputs = zip(*programs_inputs)
        programs = torch.from_numpy(np.array(programs, dtype=np.int64)).to(device, non_blocking=True)
        prog_len = programs.shape[1]

        max_inp_out_len = max(max([len(i) for i in inputs]), max([len(o) for o in outputs]))
        if seq_length is not None:
            assert seq_length <= 1024
            assert seq_length >= prog_len + max_inp_out_len , 'Fixed Batch Sequence is too small'
            max_inp_out_len = seq_length - prog_len

        assert max_inp_out_len <= 1024 - prog_len, 'Maximum input/output length is 1024'

        inputs_padded = []
        outputs_padded = []

        for i, o in zip(inputs, outputs):
            inputs_padded.append(i + [pad_idx] * (max_inp_out_len - len(i)))
            outputs_padded.append(o + [pad_idx] * (max_inp_out_len - len(o)))

        inputs_padded = torch.from_numpy(np.array(inputs_padded, dtype=np.int64)).to(device,  non_blocking=True)
        outputs_padded = torch.from_numpy(np.array(outputs_padded, dtype=np.int64)).to(device,  non_blocking=True)

        return (programs, inputs_padded), outputs_padded
    
    def get_dataloader(self, batch_size: int, seq_len: Optional[int] = None, batch_by_token_count: bool = False, device=torch.device('cpu'), pin_memory=False, shuffle=True) -> DataLoader:
        """
        batch_size: The batch size for the dataloader. 
        seq_len: If > 0, the input and output sequences will be padded to this length.
                 If <= 0, the input and output sequences will be padded to the maximum length in the dataset.
                 If None, the batches will be variable sequence length per batch.
        batch_by_token_count: If True, the dataloader will batch examples by target token count = batch_size * seq_len.
        """

        if batch_by_token_count:
            assert seq_len is not None, 'seq_len must be specified for batch_by_token_count'
            target_token_count = batch_size * seq_len
            batch_sampler = TargetTokenCountBatchSampler(self, target_token_count=target_token_count, shuffle=shuffle)
            dataloader = DataLoader(dataset=self,
                                    batch_sampler=batch_sampler,
                                    collate_fn=lambda b: self.collate_fn(b, pad_idx=self.pad_idx, seq_length=None, device=device),
                                    pin_memory=pin_memory,
                                    drop_last=False)

        else:
            if seq_len is not None and seq_len <= 0:
                seq_len = self.max_example_seq_len
                
            dataloader = DataLoader(dataset=self,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    collate_fn=lambda b: self.collate_fn(b, self.pad_idx, seq_length=seq_len, device=device),
                                    pin_memory=pin_memory,
                                    drop_last=True)

        return dataloader



class TrainingData:
    def __init__(self, augmentation_factor: int = 2, join_version: bool = False,
                training_loader: ArcTasksLoader = TRAINING_TASKLOADER,
                auxilliary_loader: List[ArcTasksLoader] = AUXILIARY_TASKLOADERS,
                num_levels = 15, seed: int = 42):
        self.augmentation_factor = augmentation_factor
        self.training_loader = training_loader
        self.auxilliary_loader = sorted(auxilliary_loader) # Ensure reproducibility
        self.join_version = join_version
        self.seed = seed
        self.num_levels = num_levels
        self.tasks = []
        self.train_examples = []
        self.test_examples = []
        self.tokenized_train_examples = []
        self.tokenized_test_examples = []
        self.program_tokenizer = None
        self.grid_tokenizer = None

    def _load_tasks_from_loader(self, loader: ArcTasksLoader):
        assert isinstance(loader, ArcTasksLoader), 'loader must be an instance of ArcTasksLoader'
        tasks = []
        if self.augmentation_factor is None or self.augmentation_factor <= 0:
            tasks.extend(loader.load_tasks(augmentation_id=None))
            return tasks
        
        for af in range(self.augmentation_factor):
            tasks.extend(loader.load_tasks(augmentation_id=af))
        return tasks
    
    def shuffle(self, lst):
        local_random = random.Random()
        local_random.seed(self.seed)
        local_random.shuffle(lst)
        return lst
    
    @staticmethod
    def trim_tasks(tasks, num_examples):
        num_examples_trimmed = 0
        trimmed_tasks = []
        for task in tasks:
            if num_examples_trimmed + len(task.train) <= num_examples:
                trimmed_tasks.append(task)
                num_examples_trimmed += len(task.train)
            else:
                break
        return trimmed_tasks

    def load_tasks(self):
        ## Load training tasks
        training_tasks = self.shuffle(self._load_tasks_from_loader(self.training_loader))

        ## Find number of training examples
        num_train_examples = sum([len(task.train) for task in training_tasks])

        ## Load auxiliary tasks
        for loader in self.auxilliary_loader:
            aux_tasks = self.trim_tasks(self.shuffle(self._load_tasks_from_loader(loader)), num_train_examples)
            training_tasks.extend(aux_tasks)

        ## Shuffle all tasks
        training_tasks = self.shuffle(training_tasks)
        sorted_tasks = sorted(training_tasks, key=lambda t: t.rank)

        ranks = [t.rank for t in sorted_tasks]
        # Added extra plus 1 because the first bin is usually empty
        quantiles = np.linspace(0, 1, (self.num_levels + 1) + 1)
        bin_edges = np.quantile(ranks, quantiles)
    
        for i in range(len(bin_edges) - 1):
            low, high = bin_edges[i], bin_edges[i+1]
            tasks = [t for t in sorted_tasks if low <= t.rank < high]
            if len(tasks) == 0:
                continue
            tasks = self.shuffle(tasks)
            self.tasks.append(tasks)
        return self.tasks

    def load_examples(self) -> None:
        for tasks in self.tasks:
            task2examples = TaskToExamples(join_version=self.join_version)
            train_examples, test_examples = [], []
            for task in tasks:
                train_egs, test_egs = task2examples(task)
                train_examples.extend(train_egs)
                test_examples.extend(test_egs)

            self.train_examples.append(self.shuffle(train_examples))
            self.test_examples.append(self.shuffle(test_examples))

        return self.train_examples, self.test_examples


    def tokenize_examples(self):
        self.program_tokenizer = ProgramTokenizer()
        self.grid_tokenizer = GridTokenizer()
        programs = [p for examples in self.train_examples for ((p, _), _) in examples]
        self.program_tokenizer.build(programs)

        def tokenize_examples(examples):
            tokenized_examples = []
            for ((p, inp), out) in examples:
                p = self.program_tokenizer.encode(p)
                inp = self.grid_tokenizer.encode(inp)
                out = self.grid_tokenizer.encode(out)
                tokenized_examples.append(((p, inp), out))
            return tokenized_examples

        # No need to shuffle as examples are already shuffled
        for examples in self.train_examples:
            self.tokenized_train_examples.append(tokenize_examples(examples))

        for examples in self.test_examples:
            self.tokenized_test_examples.append(tokenize_examples(examples))


    def load(self, cache_dir: Path = CACHE_FOLDER):
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f'training_data_{self.hash()}.pkl'
        if cache_file.exists():
            data = pickle.load(cache_file.open('rb'))
            self.tasks = data['tasks']
            self.train_examples = data['train_examples']
            self.test_examples = data['test_examples']
            self.tokenized_train_examples = data['tokenized_train_examples']
            self.tokenized_test_examples = data['tokenized_test_examples']
            self.program_tokenizer = ProgramTokenizer.from_dict(data['program_tokenizer'])
            self.grid_tokenizer = GridTokenizer.from_dict(data['grid_tokenizer'])
        else:
            self.load_tasks()
            self.load_examples()
            self.tokenize_examples()
            data = {'tasks': self.tasks,
                    'train_examples': self.train_examples,
                    'test_examples': self.test_examples,
                    'tokenized_train_examples': self.tokenized_train_examples,
                    'tokenized_test_examples': self.tokenized_test_examples,
                    'program_tokenizer': self.program_tokenizer.to_dict(),
                    'grid_tokenizer': self.grid_tokenizer.to_dict()}
            pickle.dump(data, cache_file.open('wb'))
    
        return self

    def hash(self):
        hash_args = []

        hash_args.append(self.training_loader.name)
        for loader in self.auxilliary_loader:
            hash_args.append(loader.name)

        hash_args.append(str(self.augmentation_factor))
        hash_args.append(str(self.num_levels))
        hash_args.append(str(self.join_version))
        hash_args.append(str(self.seed))

        return hash_string('_'.join(hash_args))
    
    def train_ds(self, num_levels: int, from_level: int = 0):
        assert num_levels > 0 and num_levels <= self.num_levels, f'num_levels must be between 1 and {self.num_levels}'
        assert from_level >= 0 and from_level < num_levels, f'from_level must be between 0 and num_levels({num_levels})'
        examples = []
        for i in range(from_level, num_levels):
            examples.extend(self.tokenized_train_examples[i])

        examples = self.shuffle(examples)
        return ArcExamplesDataset(examples, pad_idx=self.grid_tokenizer.PAD_IDX)
    
    def eval_ds(self, num_levels: int, from_level: int = 0):
        assert num_levels > 0 and num_levels <= self.num_levels, f'num_levels must be between 1 and {self.num_levels}'
        assert from_level >= 0 and from_level < num_levels, f'from_level must be between 0 and num_levels({num_levels})'
        examples = []
        for i in range(from_level, num_levels):
            examples.extend(self.tokenized_test_examples[i])

        examples = self.shuffle(examples)
        return ArcExamplesDataset(examples, pad_idx=self.grid_tokenizer.PAD_IDX)
# %%

# data = TrainingData().load()
# # %%
# train_ds = data.train_ds
# # %%
# train_ds[0][0][0]

# # %%
# train_ds[0][0][1]

# # %%
# len(train_ds[0][1])

# # %%
# def compute_example_len(example):
#     (p, i), o = example
#     return len(p) + max(len(i), len(o))


# sorted_indices = sorted(range(len(train_ds)), key=lambda i: compute_example_len(train_ds[i]))

# seq_batches = []
# max_seq_len = 1024
# l, r = 0, len(sorted_indices)-1
# while True:
#     # right_len is always >= left_len by design
#     batch_seq_len = 0
#     seq_batch = []
#     # Attempt to add as many right examples as possibles
#     while r > l:
#         right_idx = sorted_indices[r]
#         right_len = compute_example_len(train_ds[right_idx])
#         add_right_len = batch_seq_len + right_len 
#         if add_right_len <= max_seq_len:
#             seq_batch.append(right_idx)
#             batch_seq_len = add_right_len
#             r -= 1
#         else:
#             break

        
#     # Then attempt to add as many left examples as possible
#     while l < r:
#         left_idx = sorted_indices[l]
#         left_len = compute_example_len(train_ds[left_idx])
#         add_left_len = batch_seq_len + left_len
#         if add_left_len <= max_seq_len:
#             seq_batch.append(left_idx)
#             batch_seq_len = add_left_len
#             l += 1
#         else:
#             break

#     seq_batches.append(seq_batch)
#     if l == r:
#         break


# # %%
    
        
# seq_batches
# # %%
# batch_lens = []
# for batch in seq_batches:
#     batch_len = sum([compute_example_len(train_ds[i]) for i in batch])
#     batch_lens.append(batch_len)


# # %%
# batch_lens[-3]

# # %%
