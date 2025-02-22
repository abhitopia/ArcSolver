#%%
from typing import List, Tuple
import functools
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, BatchSampler, DataLoader
from .task import Example
from .tokenizer import ArcTokenizer, MODEL_OUTPUT, MODEL_INPUT
from .utils import get_logger
#%%

logger = get_logger()

class TargetTokenCountBatchSampler(BatchSampler):
    def __init__(self, dataset: Dataset, approx_token_count: int, shuffle: bool = True):
        self.dataset = dataset
        self.approx_token_count = approx_token_count
        assert approx_token_count >= 2000, f"target_token_count: {approx_token_count} must be greater than 3+900+900"

        self.shuffle = shuffle
        self.batches = self.create_batches()
        self.shuffle_batches()

    def shuffle_batches(self):
        if not self.shuffle:
            return
        
        # Shuffle the order of the batches
        random.shuffle(self.batches)


    def __iter__(self):
        # Shuffle the buckets and data within buckets if necessary
        if self.shuffle:
            self.shuffle_batches()

        for batch in self.batches:
            yield batch

    def __len__(self):
        # Experimentally this is only called when len(dataloader) is called
        return len(self.batches)


    @staticmethod
    def example_len(example: Example):
        inp_len = example.input.size
        out_len = example.output.size
        if inp_len == out_len:
            return (0, inp_len, out_len)
        elif inp_len < out_len:
            return (-1, out_len, inp_len)
        else:
            return (1, inp_len, out_len)


    def create_batches(self):
        # Get the sorting keys using the custom function
        dataset_keys = [self.example_len(ex) for ex in self.dataset]
        indices = list(range(len(self.dataset)))
        # Sort indices based on the custom keys
        sorted_indices = sorted(indices, key=lambda i: dataset_keys[i])
        
        batches = []
        current_batch = []
        max_inp_len = 0
        max_out_len = 0
        
        for idx in sorted_indices:
            example = self.dataset[idx]
            input_len = example.input.size
            output_len = example.output.size

            potential_max_inp_len = max(max_inp_len, input_len)
            potential_max_out_len = max(max_out_len, output_len)
            potential_batch_size = len(current_batch) + 1
            potential_batch_token_count = (potential_max_inp_len + potential_max_out_len + 3) * potential_batch_size

            if potential_batch_token_count <= self.approx_token_count:
                current_batch.append(idx)
                max_inp_len = potential_max_inp_len
                max_out_len = potential_max_out_len
            else:
                batches.append(current_batch)
                current_batch = [idx]
                max_inp_len = input_len
                max_out_len = output_len

        if current_batch:
            batches.append(current_batch)

        self.batches = batches
        return self.batches

    


    def batch_utilisation(self, batch):
        inp_lens = []
        out_lens = []
        
        if len(batch) == 0:
            return 0        
        
        for i in batch:
            example: Example = self.dataset[i]
            inp_lens.append(example.input.size)
            out_lens.append(example.output.size)

        avg_inp_len = np.mean(inp_lens)
        max_inp_len = np.max(inp_lens)
        avg_out_len = np.mean(out_lens)
        max_out_len = np.max(out_lens)

        inp_excess = (max_inp_len - avg_inp_len) * len(batch)
        out_excess = (max_out_len - avg_out_len) * len(batch)

        inp_max = max_inp_len * len(batch)
        out_max = max_out_len * len(batch)
        utilisation = 1.0 - (inp_excess + out_excess) / (inp_max + out_max)  
        return utilisation

    def stats(self):
        utils = []
        token_counts = []
        batch_lens = []
        for batch in self.batches:
            utils.append(self.batch_utilisation(batch))
            batch_lens.append(len(batch))
            max_inp_len = 0
            max_out_len = 0
            for i in batch:
                example: Example = self.dataset[i]
                max_inp_len = max(max_inp_len, example.input.size)
                max_out_len = max(max_out_len, example.output.size)
            token_counts.append((max_inp_len + max_out_len + 3) * len(batch))
            

        logger.info(f"Number of Batches: {len(self.batches)}")
        logger.info(f"Number of Examples: {sum(batch_lens)}")

        def print_stats(name, data):
            logger.info(f"-"*50)
            logger.info(f"{name} Mean: {np.mean(data)}")
            logger.info(f"{name} Median: {np.median(data)}")
            logger.info(f"{name} Max: {np.max(data)}")
            logger.info(f"{name} Min: {np.min(data)}")
            logger.info(f"{name} Std: {np.std(data)}")
        
        print_stats("Utilisation", utils)
        print_stats("Batch Size", batch_lens)
        print_stats("Token Count", token_counts)



class ArcExamplesDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer: ArcTokenizer):
        # DO NOT SHUFFLE THE EXAMPLES
        # Or the sampler will break
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
    def tokenized_example(self, idx: int) -> Tuple[MODEL_INPUT, MODEL_OUTPUT]:
        return self.collate_fn([self.examples[idx]], pad_idx=self.tokenizer.grid_tokenizer.PAD_IDX, permute=False, device=torch.device('cpu'))
    
    def subset(self, num_examples: int):
        """
        Returns a subset of the dataset
        """
        if num_examples <= 0:
            return self
        
        num_examples = min(num_examples, len(self.examples))
        indices = list(range(num_examples))
        return ArcExamplesDataset([self.examples[i] for i in indices], self.tokenizer)
    
    @staticmethod
    def collate_fn(batch: List[Example], pad_idx: int, tokenizer: ArcTokenizer, permute: False, keep_meta=True, prog_idx=None, device=torch.device('cpu'))-> Tuple[MODEL_INPUT, MODEL_OUTPUT]:
        if permute:
            batch = [ex if ex.is_original else ex.permute() for ex in batch]

        x, y = zip(*[tokenizer.encode(ex) for ex in batch])

        is_inverse = torch.stack([xi.is_inverse for xi in x], dim=0).to(device, non_blocking=True)
        cps = torch.stack([xi.color_permutation for xi in x], dim=0).to(device, non_blocking=True)
        ats = torch.stack([xi.array_transform for xi in x] , dim=0).to(device, non_blocking=True)
        prgs = [xi.program for xi in x] if prog_idx is None else [[prog_idx]] * len(x)
        prgs = torch.stack(prgs, dim=0).to(device, non_blocking=True)
        inp_grids = pad_sequence([xi.grid for xi in x], batch_first=True, padding_value=pad_idx).to(device, non_blocking=True)
        inp_indices = pad_sequence([xi.grid_indices for xi in x], batch_first=True, padding_value=-1).to(device, non_blocking=True)
        meta = [xi.meta for xi in x] if keep_meta else None

        out_grids = pad_sequence([yi.grid for yi in y], batch_first=True, padding_value=pad_idx).to(device, non_blocking=True)
        out_indices = pad_sequence([yi.grid_indices for yi in y], batch_first=True, padding_value=-1).to(device, non_blocking=True)

        target_grid = torch.cat([out_grids[:, 1:], torch.full((out_grids.size(0), 1), pad_idx, dtype=out_grids.dtype, device=device)], dim=1)

        x = MODEL_INPUT(
            is_inverse=is_inverse,
            color_permutation=cps,
            array_transform=ats,
            program=prgs,
            grid=inp_grids,
            grid_indices=inp_indices,
            meta=meta
        )

        y = MODEL_OUTPUT(
            grid=out_grids,
            grid_indices=out_indices,
            target_grid=target_grid
        )

        return x, y        
    
    def get_dataloader(self,
                    token_count: int,
                    device=torch.device('cpu'), 
                    pin_memory: bool=True, 
                    shuffle: bool=True,
                    num_workers: int = 4,
                    permute: bool = False
                    ) -> DataLoader:
        """
        Returns a DataLoader for the dataset
        """

        pad_idx = self.tokenizer.grid_tokenizer.PAD_IDX

        collate_fn = functools.partial(ArcExamplesDataset.collate_fn,
                                    tokenizer=self.tokenizer,
                                    pad_idx=pad_idx, 
                                    device=device,
                                    keep_meta=True, 
                                    permute=permute)

        batch_sampler = TargetTokenCountBatchSampler(self, approx_token_count=token_count, shuffle=shuffle)
        dl = DataLoader(dataset=self,
                        batch_sampler=batch_sampler,
                        collate_fn=collate_fn,
                        pin_memory=pin_memory,
                        num_workers=num_workers,
                        drop_last=False)

        return dl
    
# %%
# %%
