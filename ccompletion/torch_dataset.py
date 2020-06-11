from torch.utils.data import IterableDataset
from transformers import GPT2TokenizerFast

import codecs
import math
import os
import torch


class MyDataset(IterableDataset):
    def __init__(self, dataset_dir='repositories', max_seq_len=1024):
        super(MyDataset).__init__()

        # initialize dataset directory path
        assert os.path.isdir(dataset_dir)
        self.dataset_dir = dataset_dir

        # initialize dataset tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.max_seq_len = max_seq_len

    def __iter__(self):
        for root, _, files in os.walk(self.dataset_dir):
            for pf in files:
                # skip files that are not python src codes
                if not pf.endswith('.py'):
                    continue

                # open each src file and collate all unique tokens
                pf_path = os.path.join(root, pf)
                with codecs.open(pf_path, 'r', 'utf8', errors='replace') as fd:
                    code = fd.read() + '<|endoftext|>'
                    chunks = self.__to_chunks(code)
                    for chunk in chunks:
                        yield torch.tensor(chunk, dtype=torch.long)

    def __to_chunks(self, code):
        tokens = self.tokenizer.encode(code)
        n_tokens = len(tokens)

        # divide longer tokens into chunks if it exceeds max_seq_len
        if n_tokens > self.max_seq_len:
            chunks_length = math.ceil(n_tokens / self.max_seq_len)

            for j in range(chunks_length):
                start_index = j * self.max_seq_len
                end_index = (j+1) * self.max_seq_len

                token_chunk = tokens[start_index: end_index]
                yield token_chunk
        else:
            yield tokens
