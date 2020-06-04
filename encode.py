from tqdm import tqdm

from ccompletion.encoder import get_encoder

import codecs
import fire
import numpy as np
import os


def get_total_file_count(dataset_dir):
    total = 0
    for _, _, files in os.walk(dataset_dir):
        total += len(files)

    return total


def encode(dataset_dir='repositories', token_count_threshold=10,
           redo=False, output_file='dataset.npz'):
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
    encoder = get_encoder()
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
                    src = fd.read()
                    tokens = encoder.encode(src, add_start_token=True)

                    # ignore files that have tokens of length < threshold
                    if len(tokens) < token_count_threshold:
                        ignored_files.append(pf_path)
                        pbar.update()
                        continue

                    token_chunks.append(np.stack(tokens))
                    pbar.update()

    print('INFO - Ignored files:')
    for f in ignored_files:
        print('     -', f)

    # save encoded tokens in a compressed format using numpy
    np.savez_compressed(output_file, token_chunks=token_chunks)


if __name__ == '__main__':
    fire.Fire(encode)
