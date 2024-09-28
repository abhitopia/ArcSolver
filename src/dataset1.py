#%%
from typing import List, Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler, DataLoader
from .task1 import Example
from .tokenizer import ArcTokenizer, MODEL_OUTPUT, MODEL_INPUT


class TargetTokenCountBatchSampler(BatchSampler):
    def __init__(self, dataset: Dataset, approx_token_count: int, min_util: float = 0.80, shuffle: bool = True):
        self.dataset = dataset
        self.approx_token_count = approx_token_count
        assert approx_token_count >= 2000, f"target_token_count: {approx_token_count} must be greater than 3+900+900"

        self.shuffle = shuffle
        self.min_util = min_util
        self._dataset_lens = [self.sample_len(x, y) for x, y in self.dataset]
        self.batches = self.create_batches()
        self.shuffle_batches()

    def shuffle_batches(self):
        if not self.shuffle:
            return
        # Shuffle the order of the data within each batch
        for batch in self.batches:
            random.shuffle(batch)
        # Shuffle the order of the batches
        random.shuffle(self.batches)

    @staticmethod
    def sample_len(model_input: MODEL_INPUT, model_output: MODEL_OUTPUT):
        inp_len = len(model_input.grid)
        # print(inp_len, model_input.input.size)
        out_len = len(model_output.grid)
        if inp_len == out_len:
            return (0, inp_len, out_len)
        elif inp_len < out_len:
            return (-1, out_len, inp_len)
        else:
            return (1, inp_len, out_len)
        
    def merge_batches(self, batches, token_counts, batch_widths):
        sorted_batch_indices = sorted(range(len(batches)), key=lambda i: (token_counts[i], batch_widths[i]), reverse=False)

        super_batches = []

        super_batch = []

        super_batch_width = 0
        super_batch_widths = []

        super_batch_token_count = 0
        super_batch_token_counts = []

        for i in sorted_batch_indices:
            if token_counts[i] >= self.approx_token_count:
                super_batches.append(batches[i])
                super_batch_widths.append(batch_widths[i])
                super_batch_token_counts.append(token_counts[i])
                continue

            inc_batch_width = max(batch_widths[i], super_batch_width)
            inc_batch_token_count = inc_batch_width * (len(batches[i]) + len(super_batch))

            if inc_batch_token_count <= self.approx_token_count:
                super_batch += batches[i]
                super_batch_width = inc_batch_width
                super_batch_token_count = inc_batch_token_count
            elif super_batch_token_count <= 0.5*self.approx_token_count:
                # skip current batch and move to next
                super_batches.append(batches[i])
                super_batch_widths.append(batch_widths[i])
                super_batch_token_counts.append(token_counts[i])
            else:
                super_batches.append(super_batch)
                super_batch_widths.append(super_batch_width)
                super_batch_token_counts.append(super_batch_token_count)
                super_batch = batches[i]
                super_batch_width = batch_widths[i]
                super_batch_token_count = token_counts[i]


        if len(super_batch) > 0:
            super_batches.append(super_batch)
            super_batch_widths.append(super_batch_width)
            super_batch_token_counts.append(super_batch_token_count)

        return super_batches, super_batch_token_counts, super_batch_widths


    def reduce_num_batches(self, batches, batch_token_counts, batch_widths):
        # print("Reducing number of batches")
        mean_util = np.mean([self.batch_utilisation(batch) for batch in batches])

        score = mean_util/len(batches)
        for _ in range(10):
            new_batches, new_batch_token_counts, new_batch_widths = self.merge_batches(batches, batch_token_counts, batch_widths)
            mean_util = np.mean([self.batch_utilisation(batch) for batch in new_batches])
            new_score = mean_util/len(new_batches)
            # print(new_score, len(new_batches))
            if new_score > score:
                batches = new_batches
                batch_token_counts = new_batch_token_counts
                batch_widths = new_batch_widths
                score = new_score
            else:
                break

        return new_batches



    def create_batches(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        sorted_indices = sorted(indices, key=lambda i: self._dataset_lens[i], reverse=False)

        batches = [] 
        batch = []
        avg_batch_inp_len = 0
        avg_batch_out_len = 0
        max_batch_inp_len = 0
        max_batch_out_len = 0
        batch_token_counts = []
        batch_token_count = 0
        batch_widths = []


        for idx in sorted_indices:
            x, y = self.dataset[idx]
            input_len = len(x.grid)
            output_len = len(y.grid)

            # Incremental stats
            inc_max_inp_len = max(input_len, max_batch_inp_len)
            inc_max_out_len = max(output_len, max_batch_out_len)

            inc_avg_inp_len = (avg_batch_inp_len * len(batch) + input_len) / (len(batch) + 1)
            inc_avg_out_len = (avg_batch_out_len * len(batch) + output_len) / (len(batch) + 1)

            inc_inp_excess = (inc_max_inp_len - inc_avg_inp_len)
            inc_out_excess = (inc_max_out_len - inc_avg_out_len)

            inc_batch_util = 1.0 - ((inc_inp_excess + inc_out_excess) / (inc_max_inp_len + inc_max_out_len))
            inc_batch_token_count = (inc_max_inp_len + inc_max_out_len + 3) * (len(batch) + 1)

            if inc_batch_token_count >= self.approx_token_count or inc_batch_util < self.min_util:
                batches.append(batch)
                batch_token_counts.append(batch_token_count)
                batch_widths.append(max_batch_inp_len + max_batch_out_len + 3)
                batch = [idx]
                avg_batch_inp_len = input_len
                avg_batch_out_len = output_len
                max_batch_inp_len = input_len
                max_batch_out_len = output_len
                batch_token_count = input_len + output_len + 3
            else:
                batch.append(idx)
                avg_batch_inp_len = inc_avg_inp_len
                avg_batch_out_len = inc_avg_out_len
                max_batch_inp_len = inc_max_inp_len
                max_batch_out_len = inc_max_out_len
                batch_token_count = inc_batch_token_count 
    
        if len(batch) > 0:
            batches.append(batch)
            batch_token_counts.append(batch_token_count)
            batch_widths.append(max_batch_inp_len + max_batch_out_len + 3)

        batches = self.reduce_num_batches(batches, batch_token_counts, batch_widths)

        return batches


    def batch_utilisation(self, batch):
        inp_lens = []
        out_lens = []
        
        if len(batch) == 0:
            return 0        

        for i in batch:
            x, y = self.dataset[i]
            inp_lens.append(len(x.grid))
            out_lens.append(len(y.grid))

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

    
    def __iter__(self):
        # Shuffle the buckets and data within buckets if necessary
        if self.shuffle:
            self.shuffle_batches()

        for batch in self.batches:
            yield batch


    def __len__(self):
        # Experimentally this is only called when len(dataloader) is called
        return len(self.batches)


    def stats(self):
        utils = []
        token_counts = []
        batch_lens = []
        for batch in self.batches:
            utils.append(self.batch_utilisation(batch))
            batch_lens.append(len(batch))
            seq_len = 0
            for i in batch:
                x, y = self.dataset[i]
                seq_len = max(len(x.grid) + len(y.grid) + 3, seq_len)
            token_counts.append(seq_len * len(batch))
            

        print("Number of Batches: ", len(self.batches))
        print("Number of Examples: ", sum(batch_lens))

        def print_stats(name, data):
            print(f"-"*50)
            print(f"{name} Mean: {np.mean(data)}")
            print(f"{name} Median: {np.median(data)}")
            print(f"{name} Max: {np.max(data)}")
            print(f"{name} Min: {np.min(data)}")
            print(f"{name} Std: {np.std(data)}")
        
        print_stats("Utilisation", utils)
        print_stats("Batch Size", batch_lens)
        print_stats("Token Count", token_counts)


class ArcExamplesDataset(Dataset):
    def __init__(self, examples: List[Example], tokenizer: ArcTokenizer):

        # DO NOT SHUFFLE THE EXAMPLES
        # Or the sampler will break
        self.examples = examples
        self.tokenizer = tokenizer
        self.x, self.y = zip(*[self.tokenizer.encode(e) for e in examples])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
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
    def collate_fn(batch: List[Tuple[MODEL_INPUT, MODEL_OUTPUT]], pad_idx: int, device=torch.device('cpu'))-> Tuple[MODEL_INPUT, MODEL_OUTPUT]:
        x, y  = zip(*batch)

        cps = [xi.color_permutation for xi in x]
        ats = [xi.array_transform for xi in x]
        prgs = [xi.program for xi in x]
        inp_grids = [xi.grid for xi in x]
        inp_indices = [xi.grid_indices for xi in x]
        out_grids = [yi.grid for yi in y]
        out_indices = [yi.grid_indices for yi in y]
        meta = [xi.meta for xi in x]

        prgs = torch.tensor(prgs, dtype=torch.long).to(device, non_blocking=True)
        cps = torch.tensor(cps, dtype=torch.long).to(device, non_blocking=True)
        ats = torch.tensor(ats, dtype=torch.long).to(device, non_blocking=True)

        inp_seq_len = max([len(i) for i in inp_grids])
        out_seq_len = max([len(o) for o in out_grids])

        inp_grids = [i + [pad_idx] * (inp_seq_len - len(i)) for i in inp_grids]
        out_grids = [o + [pad_idx] * (out_seq_len - len(o)) for o in out_grids]
        inp_indices = [i + [(-1, -1)] * (inp_seq_len - len(i)) for i in inp_indices]
        out_indices = [o + [(-1, -1)] * (out_seq_len - len(o)) for o in out_indices]

        inp_grids = torch.tensor(inp_grids, dtype=torch.long).to(device, non_blocking=True)
        out_grids = torch.tensor(out_grids, dtype=torch.long).to(device, non_blocking=True)
        inp_indices = torch.tensor(inp_indices, dtype=torch.long).to(device, non_blocking=True)
        out_indices = torch.tensor(out_indices, dtype=torch.long).to(device, non_blocking=True)

        target_grid = torch.cat([out_grids[:, 1:], torch.full((out_grids.size(0), 1), pad_idx, dtype=out_grids.dtype)], dim=1)

        x = MODEL_INPUT(
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
                    min_util: float=0.70,
                    ) -> DataLoader:
        """
        batch_size: The batch size for the dataloader. 
        seq_len: If > 0, the input and output sequences will be padded to this length.
                 If <= 0, the input and output sequences will be padded to the maximum length in the dataset.
                 If None, the batches will be variable sequence length per batch.
        batch_by_token_count: If True, the dataloader will batch examples by target token count = batch_size * seq_len.
        """

        pad_idx = self.tokenizer.grid_tokenizer.PAD_IDX

        batch_sampler = TargetTokenCountBatchSampler(self, approx_token_count=token_count, min_util=min_util, shuffle=shuffle)
        dl = DataLoader(dataset=self,
                                batch_sampler=batch_sampler,
                                collate_fn=lambda b: self.collate_fn(b, pad_idx, device=device),
                                pin_memory=pin_memory,
                                drop_last=False)

        return dl
    
# %%
