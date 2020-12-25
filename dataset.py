from collections import defaultdict
from pathlib import Path
from torch.utils.data import IterableDataset, Dataset
from transformers import GPT2TokenizerFast, T5Tokenizer
from typing import Callable, List, Union

import fire
import math
import pickle
import os


class PythonReposDataset(IterableDataset):
    def __init__(
        self,
        dataset_dir: str = 'repositories',
        count_fn: Callable[[str], int] = None,
        batch_size: int = 8,
        block_size: int = 2 << 7,
        return_str: bool = False,
    ):
        self.count_fn = count_fn
        self.dataset_path = Path(dataset_dir)
        self.batch_size = batch_size
        self.block_size = block_size
        self.return_str = return_str

        assert count_fn is not None
        assert self.dataset_path.is_dir()

        self.source_files = list(self.dataset_path.rglob('*.py'))
        self.buckets = defaultdict(list)

    def __iter__(self) -> List[str]:
        # send batches of width <= `block_size`
        batch = []
        for i in range(len(self.source_files)):
            # some directories also have a '.py' ending
            # we need actual .py source files so we skip directories
            if not self.source_files[i].is_file():
                continue

            # read contents of the file
            src = self.read_contents(i).strip()

            # we don't include empty python source files
            if not src:
                continue

            try:
                # count tokens of the current source file
                count = self.count_fn(src)
            except ValueError:
                # is raised if the tokens contain unknown tokens
                # it's better to skip source files with unknown tokens
                continue

            # append source file contents to the current batch
            # if its token length is less than `block_size`
            if count < self.block_size:
                src_path = str(self.source_files[i])
                batch.append(src if self.return_str else src_path)
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
                yield [
                    self.read_contents(j)
                    if self.return_str
                    else str(self.source_files[j])
                    for j in batch
                ]

    def read_contents(self, index: int) -> str:
        path = self.source_files[index]
        with path.open('r', encoding='utf-8') as fd:
            return fd.read().strip()


class PythonReposCachedDataset(Dataset):
    def __init__(self, cache: Union[str, Path]):
        with open(cache, 'rb') as fd:
            self.batches = pickle.load(fd)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        # some files may have been removed after unzipping (very rare)
        # don't include empty files
        batch = self.batches[index]
        results = []

        for path in batch:
            if not os.path.isfile(path):
                continue

            contents = self.read_contents(path).strip()
            if not contents:
                continue

            results.append(contents)

        # every batch should contain atleast one item
        if not results:
            raise AssertionError(
                f"Batch {batch} doesn't contain any items.")

        return results

    def read_contents(_, path: str) -> str:
        with open(path, 'r', encoding='utf-8') as fd:
            return fd.read().strip()


def preencode(variant: str, dataset_dir: str = 'repositories', batch_size: int = 8, block_size: int = 2 << 7):
    dataset = PythonReposDataset(
        dataset_dir=dataset_dir,
        count_fn=get_count_fn(variant),
        batch_size=batch_size,
        block_size=block_size,
        return_str=False,
    )

    # iterate undeterminate dataset
    # to become determinate
    batches = []
    for batch in dataset:
        batches.append(batch)

    # pickle batches for training use
    path = f'_{variant}_{batch_size}_{block_size}.pkl'
    with open(path, 'wb') as fd:
        pickle.dump(batches, fd)
        print(f'Dumped batches as "{path}"')


def get_count_fn(variant: str) -> Callable[[str], int]:
    if variant == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        return create_count_fn(tokenizer)

    if variant == 't5':
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        return create_count_fn(tokenizer)

    raise AssertionError(f'Unknown tokenizer variant "{variant}"')


def create_count_fn(tokenizer):
    def count_fn(src):
        tokens = tokenizer.encode(src)
        if tokenizer.unk_token_id in tokens:
            raise ValueError(f'src file contains unknown tokens')
        return len(tokens)

    return count_fn


if __name__ == '__main__':
    fire.Fire(preencode)
