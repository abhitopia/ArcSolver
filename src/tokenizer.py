from dataclasses import dataclass
import re
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
from collections import namedtuple
import json
import numpy as np
import torch
from .task1 import ArrayTransform, ColorPermutation, Example


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
        self.BOS_TOKEN = '[['
        self.EOS_TOKEN = ']]'
        self.NEW_ROW_TOKEN = '['

        tokens = [self.PAD_TOKEN, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[[', ']', '[', ']]']
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        super().__init__(token2idx=token2idx, idx2token=idx2token, frozen=True)
        self.PAD_IDX = self.token2idx[self.PAD_TOKEN]
        self.BOS_IDX = self.token2idx[self.BOS_TOKEN]
        self.EOS_IDX = self.token2idx[self.EOS_TOKEN]
        self.NEW_ROW_IDX = self.token2idx[self.NEW_ROW_TOKEN]
        assert self.PAD_IDX == 0

    def decode(self, sequence, remove_padding=True):
        tokens = super().decode(sequence)
        if remove_padding:
            tokens = [token for token in tokens.split(' ') if token != self.PAD_TOKEN]
            return ' '.join(tokens)
    
        return tokens



class GridSerializer:
    @staticmethod
    def serialize_array(array: np.ndarray) -> str:
        list_of_lists = array.tolist()
        array_str = str(list_of_lists)
        array_str = array_str.replace('],' , ' ],').replace('[[', '[[ ').replace(']]', ' ]]').replace(' [', ' [ ').replace(',', '')
        num_tokens = len(array_str.split())
        assert num_tokens == array.shape[0] * (array.shape[1] + 2)
        indices = GridSerializer.indices(array)
        assert num_tokens == len(indices)
        return array_str, indices

    @staticmethod
    def indices(array: np.ndarray) -> List[Tuple[int, int]]:
        height, width = array.shape
        indices = np.indices((height, width + 2)).transpose(1, 2, 0)
        indices = indices.reshape(height*(width+2), 2)
        indices = [tuple(row) for row in indices]
        return indices

    @staticmethod
    def deserialize_array(array_str: str) -> np.ndarray:
        pattern = r'(?<=\d) (?=\d)'
        replacement = ', '
        result = re.sub(pattern, replacement, array_str)
        result = result.replace('] [', '], [')
        result = json.loads(result)
        return np.array(result)


class ProgramTokenizer(Tokenizer):
    def __init__(self):
        super().__init__(frozen=False)

    def build(self, tokens: List[str]):
        if self.frozen:
            raise ValueError('Tokenizer is frozen. No new tokens can be added.')
        for token in tokens:
            for t in token.strip().split(' '):
                if len(t) == 1:
                    print(f'Adding token: {token}')
                self.add_token(t)
        self.frozen = True


class ColorPermutationTokenizer(Tokenizer):
    def __init__(self):
        tokens = [cp.name for cp in list(ColorPermutation)]
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        super().__init__(token2idx=token2idx, idx2token=idx2token, frozen=True)


class ArrayTransformTokenizer(Tokenizer):
    def __init__(self):
        tokens = [at.name for at in list(ArrayTransform)]
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        super().__init__(token2idx=token2idx, idx2token=idx2token, frozen=True)


class MODEL_INPUT(NamedTuple):
    color_permutation: torch.Tensor
    array_transform: torch.Tensor
    program: torch.Tensor
    grid: torch.Tensor
    grid_indices: torch.Tensor
    meta: Optional[List[Dict[str, str]]]


class MODEL_OUTPUT(NamedTuple):
    grid: torch.Tensor
    grid_indices: torch.Tensor
    target_grid: Optional[torch.Tensor]


class ArcTokenizer:
    def __init__(self) -> None:
        self.grid_tokenizer = GridTokenizer()
        self.program_tokenizer = ProgramTokenizer()
        self.color_permutation_tokenizer = ColorPermutationTokenizer()
        self.array_transform_tokenizer = ArrayTransformTokenizer()

    def encode(self, example: Example) -> Tuple[MODEL_INPUT, MODEL_OUTPUT]:
        input_grid_ser, input_indices = GridSerializer.serialize_array(example.input)
        input_grid_encoded = self.grid_tokenizer.encode(input_grid_ser)
        output_grid_ser, output_indices = GridSerializer.serialize_array(example.output)
        output_grid_encoded = self.grid_tokenizer.encode(output_grid_ser)
        program_encoded = self.program_tokenizer.encode(example.program_id)
        color_permutation_encoded = self.color_permutation_tokenizer.encode(example.color_perm)
        array_transform_encoded = self.array_transform_tokenizer.encode(example.transform)

        x = MODEL_INPUT(
            color_permutation = color_permutation_encoded,
            array_transform = array_transform_encoded,
            program = program_encoded,
            grid = input_grid_encoded,
            grid_indices = input_indices,
            meta={'task_id': example.task_id, 'example_id': example.idx, 'dataset': example.dataset}
        )
        y = MODEL_OUTPUT(
            grid = output_grid_encoded,
            grid_indices = output_indices,
            target_grid = output_grid_encoded[:-1] + [self.grid_tokenizer.PAD_IDX]
            )
        return x, y

    def decode(self, x: MODEL_INPUT, y: MODEL_OUTPUT=None) -> Example:
        input_decoded = self.grid_tokenizer.decode(x.grid)
        output_decoded = self.grid_tokenizer.decode(y.grid) if y else None
        program_decoded = self.program_tokenizer.decode(x.program)
        color_permutation_decoded = self.color_permutation_tokenizer.decode(x.color_permutation)
        array_transform_decoded = self.array_transform_tokenizer.decode(x.array_transform)

        example = Example(
            idx=x.meta['example_id'],
            task_id=x.meta['task_id'],
            dataset=x.meta['dataset'],
            input=GridSerializer.deserialize_array(input_decoded),
            output=GridSerializer.deserialize_array(output_decoded),
            program_id=program_decoded,
            color_perm=color_permutation_decoded,
            transform=array_transform_decoded
        )

        return example
    
    
    def to_dict(self):
        assert self.program_tokenizer.frozen, 'ProgramTokenizer must be frozen before saving.'
        return {
            'color_permutation_tokenizer': self.color_permutation_tokenizer.to_dict(),
            'array_transform_tokenizer': self.array_transform_tokenizer.to_dict(),
            'program_tokenizer': self.program_tokenizer.to_dict(),
            'grid_tokenizer': self.grid_tokenizer.to_dict(),
        }
    
    def build_program_tokenizer(self, examples: List[Example]):
        programs = [example.program_id for example in examples]
        self.program_tokenizer.build(programs)
    
    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.color_permutation_tokenizer = ColorPermutationTokenizer.from_dict(data['color_permutation_tokenizer'])
        obj.array_transform_tokenizer = ArrayTransformTokenizer.from_dict(data['array_transform_tokenizer'])
        obj.program_tokenizer = ProgramTokenizer.from_dict(data['program_tokenizer'])
        obj.grid_tokenizer = GridTokenizer.from_dict(data['grid_tokenizer'])
        return obj
    