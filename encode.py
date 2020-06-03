from tqdm import tqdm

from ccompletion.encoder import get_encoder

import codecs
import numpy as np
import os

DATASET_DIR = 'repositories'


def get_total_file_count():
    total = 0
    for _, _, files in os.walk(DATASET_DIR):
        total += len(files)

    return total


def main():
    # assert directory "repositories" exists
    if not os.path.isdir(DATASET_DIR):
        print('ERROR - Directory "repositories" not found.')
        print('Download the dataset first by running download_dataset.py')
        exit(1)

    encoder = get_encoder()
    token_chunks = []
    n_files = get_total_file_count()

    with tqdm(total=n_files) as t:
        for root, _, files in os.walk(DATASET_DIR):
            for pf in tqdm(files):
                # skip files that are not python src codes
                if not pf.endswith('.py'):
                    continue

                # open each src file and collate all unique tokens
                pf_path = os.path.join(root, pf)
                with codecs.open(pf_path, 'r', 'utf-8') as fd:
                    src = fd.read()
                    tokens = encoder.encode(src, add_start_token=True)

                    # ignore files that only have 1 or less token(s)
                    if len(tokens) <= 1:
                        print('INFO - Will ignore {}'.format(pf_path))
                        continue

                    token_chunks.append(np.stack(tokens))

                # update tqdm progress
                t.update()

    # save encoded tokens in a compressed format using numpy
    np.savez_compressed('dataset.npz', token_chunks=token_chunks)


if __name__ == '__main__':
    main()
