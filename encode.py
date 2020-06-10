from tqdm import tqdm

from ccompletion.encoder import get_encoder
from ccompletion.hparams import get_hparams

import codecs
import fire
import math
import os
import torch


def encode(dataset_dir='repositories', token_count_threshold=10,
           output_file='dataset.pt', frequency_threshold=20, redo=False):
    """
    Pre-encodes all of the files in the `dataset_dir` directory.
    This uses the token IDs provided by `ccompletion.encoder`. This module
    will create `encoder.json` if it doesn't exist yet. This will allow
    model training iterations much faster since no further processing is
    needed after this step.

    Parameters:
    dataset_dir (string): the directory whose files will be encoded
    token_count_threshold (int): files whose token count is less than this threshold
        will be ignored and won't be included in the training
    output_file (string): the name of the serialized output file
    frequency_threshold (int): tokens whose frequency is below this threshold will
        not be treated as unique and will be encoded as `<|unknown|>`
    redo (boolean): flag that determines whether to redo this encoding process
        even if `output_file` already exists
    """
    # assert directory "repositories" exists
    if not os.path.isdir(dataset_dir):
        print('ERROR - Directory "repositories" not found.')
        print('Download the dataset first by running download_dataset.py')
        exit(1)

    if os.path.isfile(output_file) and not redo:
        print('INFO - {} already exists.'.format(output_file))
        print('INFO - Pass --redo=True to redo encoding.')
        exit(0)

    total_file_count = get_total_file_count(dataset_dir)
    encoder = get_encoder(threshold=frequency_threshold)
    max_seq_len = get_hparams()['n_ctx'] + 1
    token_chunks = []
    ignored_files = []

    print('INFO - Encoding source files...')
    with tqdm(total=total_file_count) as pbar:
        for root, _, files in os.walk(dataset_dir):
            for pf in files:
                # skip files that are not python src codes
                if not pf.endswith('.py'):
                    pbar.update()
                    continue

                # open each src file and collate all unique tokens
                pf_path = os.path.join(root, pf)
                with codecs.open(pf_path, 'r', 'utf8', errors='replace') as fd:
                    # read source code of current file fd
                    src = fd.read()

                    # tokenize source code of the current file
                    tokens = encoder.encode(src, add_start_token=True)
                    n_tokens = len(tokens)

                    # ignore files that have tokens of length < threshold
                    if n_tokens < token_count_threshold:
                        ignored_files.append(pf_path)
                        pbar.update()
                        continue

                    # divide longer tokens into chunks if it exceeds max_seq_len
                    if n_tokens > max_seq_len:
                        chunks_length = math.ceil(n_tokens / max_seq_len)

                        for j in range(chunks_length):
                            start_index = j * max_seq_len
                            end_index = (j+1) * max_seq_len

                            token_chunk = tokens[start_index: end_index]
                            token_chunk = pad(token_chunk, max_seq_len)
                            token_chunks.append(token_chunk)
                    else:
                        tokens = pad(tokens, max_seq_len)
                        token_chunks.append(tokens)

                    # update tqdm progress for this file
                    pbar.update()

    # print list of ignored files (if not empty)
    if ignored_files:
        print('INFO - Ignored files:')
        for f in ignored_files:
            print('     -', f)

    # save encoded tokens as a PyTorch tensor
    token_chunks = torch.Tensor(token_chunks)
    torch.save(token_chunks, output_file)
    print('INFO - Encoded dataset saved in {}'.format(output_file))


def pad(arr, max_seq_len, value=0):
    """
    Pads the given `arr` with values `value`.

    Parameters:
    arr (array): the array to be padded
    max_seq_len (int): the length of the array after padding
    value (int): the value to be inserted when padding

    Returns:
    padded (array): the padded array
    """
    # no need to pad array if its length is already the maximum allowed
    if len(arr) >= max_seq_len:
        return arr

    remainder = max_seq_len - len(arr)
    return arr + [value] * remainder


def get_total_file_count(dataset_dir):
    """
    Returns the total number of files inside `dataset_dir`.

    Parameters:
    dataset_dir (string): the directory whose files will be counted recursively

    Returns:
    total (int): the total number of files
    """
    total = 0
    for _, _, files in os.walk(dataset_dir):
        total += len(files)

    return total


if __name__ == '__main__':
    fire.Fire(encode)
