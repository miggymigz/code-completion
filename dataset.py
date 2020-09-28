from pathlib import Path
from torch.utils.data import IterableDataset

import math
import torch


class PythonReposDataset(IterableDataset):
    def __init__(self, dataset_dir: str = 'repositories'):
        # initialize dataset directory path
        self.dataset_path = Path(dataset_dir)
        assert self.dataset_path.is_dir()

    def __iter__(self):
        for f in self.dataset_path.rglob('*.py'):
            with f.open('r', encoding='utf-8') as fd:
                code = fd.read().strip()
                yield code

    def __len__(self):
        return len(list(self.dataset_path.rglob('*.py')))
