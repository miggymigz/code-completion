from collections import defaultdict
from pathlib import Path
from torch.utils.data import IterableDataset
from typing import Callable

import math


class PythonReposDataset(IterableDataset):
    def __init__(
        self,
        dataset_dir: str = 'repositories',
        count_fn: Callable[[str], int] = None,
        batch_size: int = 8,
        block_size: int = 2 << 7,
    ):
        self.count_fn = count_fn
        assert count_fn is not None

        self.dataset_path = Path(dataset_dir)
        assert self.dataset_path.is_dir()

        self.batch_size = batch_size
        self.block_size = block_size
        self.source_files = list(self.dataset_path.rglob('*.py'))

        # cache for storing filenames and their corresponding token counts
        self.buckets = defaultdict(list)

    def __iter__(self) -> str:
        # send batches of width <= `block_size`
        batch = []
        for i in range(len(self.source_files)):
            # count tokens of the current source file
            src = self.read_contents(i)
            count = self.count_fn(src)

            # append source file contents to the current batch
            # if its token length is less than `block_size`
            if count < self.block_size:
                batch.append(src)
            # store count of current source file in a bucket
            # in preparation for later dispatching
            else:
                bucket_id = math.ceil(count / self.block_size) - 1
                self.buckets[bucket_id].append(i)

            # return batch if it is already complete
            # also, empty list in preparation for the next batch
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        # send remaining batches (inside buckets)
        for _, indices in self.buckets.items():
            n_batches = math.ceil(len(indices) / self.batch_size)
            for i in range(n_batches):
                batch = indices[i * self.batch_size: i *
                                self.batch_size + self.batch_size]
                yield [self.read_contents(j) for j in batch]

    def read_contents(self, index) -> str:
        f = self.source_files[index]
        with f.open('r', encoding='utf-8') as fd:
            return fd.read().strip()
