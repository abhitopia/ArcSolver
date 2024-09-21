from typing import List
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
        tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[[', '],[', ']]', self.PAD_TOKEN]
        token2idx = {token: idx for idx, token in enumerate(tokens)}
        idx2token = {idx: token for idx, token in enumerate(tokens)}
        super().__init__(token2idx=token2idx, idx2token=idx2token, frozen=True)
        self.PAD_IDX = self.token2idx[self.PAD_TOKEN]

    def decode(self, sequence, remove_padding=True):
        tokens = super().decode(sequence)
        if remove_padding:
            tokens = [token for token in tokens.split(' ') if token != self.PAD_TOKEN]
            return ' '.join(tokens)
    
        return tokens


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


class ArcTokenizer:
    def __init__(self) -> None:
        self.grid_tokenizer = GridTokenizer()
        self.program_tokenizer = ProgramTokenizer()
        self.color_permutation_tokenizer = ColorPermutationTokenizer()
        self.array_transform_tokenizer = ArrayTransformTokenizer()
    
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
    