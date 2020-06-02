from tqdm import tqdm

from ccompletion.encoder import get_encoder

import codecs
import numpy as np
import os

DATASET_DIR = 'repositories'


def main():
    # assert directory "repositories" exists
    if not os.path.isdir(DATASET_DIR):
        print('ERROR - Directory "repositories" not found.')
        print('Download the dataset first by running download_dataset.py')
        exit(1)

    encoder = get_encoder()
    token_chunks = []

    for root, _, files in os.walk(DATASET_DIR):
        for pf in tqdm(files):
            # skip files that are not python src codes
            if not pf.endswith('.py'):
                continue

            # open each src file and collate all unique tokens
            with codecs.open(os.path.join(root, pf), 'r', 'utf-8') as fd:
                src = fd.read()
                tokens = encoder.encode(src, add_start_token=True)
                token_chunks.append(np.stack(tokens))

    # save encoded tokens in a compressed format using numpy
    np.savez_compressed('dataset.npz', token_chunks=token_chunks)


if __name__ == '__main__':
    main()
