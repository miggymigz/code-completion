from ccompletion.tokenizer import PythonTokenizer
from tqdm import tqdm

import codecs
import fire
import os
import pickle
import tokenize as ptokenize


def encode(
    dataset_dir='repositories',
    output_file='dataset.txt',
    redo=False
):
    """
    Pre-encodes all of the files in the `dataset_dir` directory.
    This uses the token IDs provided by `ccompletion.encoder`. This module
    will create `encoder.json` if it doesn't exist yet. This will allow
    model training iterations much faster since no further processing is
    needed after this step.

    Parameters:
    dataset_dir (string): the directory whose files will be encoded
    output_file (string): the name of the serialized output file
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
    tokenizer = PythonTokenizer(vocab_file='models/vocab.bpe')

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
                    src = fd.read().strip()

                    # ignore files that are empty
                    if not src:
                        pbar.write(f'INFO - Empty file: {pf_path}')
                        pbar.update()
                        continue

                    # tokenize source code of the current file
                    try:
                        tokens = tokenizer.encode(src)
                    except (ptokenize.TokenError, IndentationError, SyntaxError):
                        # ignore python files that could not be tokenized
                        # as they may be used by test files e.g. (google/pytype/tokenerror1.py)
                        # this way, our dataset will only contain grammatical python source files
                        pbar.write(f'INFO - Malformed file: {pf_path}')
                    except (LookupError, UnicodeDecodeError):
                        # some python files in the repositories use encodings other than
                        pbar.write(f'INFO - Unsupported encoding: {pf_path}')
                    else:
                        # pickle dump current tokens
                        with open(output_file, 'ab') as fd:
                            pickle.dump(tokens, fd)
                    finally:
                        pbar.update()

    print(f'INFO - Encoded dataset saved in {output_file}')


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


def iterpickle(fname):
    with open(fname, 'rb') as fd:
        while fd.peek(1):
            yield pickle.load(fd)


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
