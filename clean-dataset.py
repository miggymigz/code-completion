from pathlib import Path
from tqdm import tqdm
from yapf.yapflib.yapf_api import FormatFile

import ast
import fire
import lib2to3


def clean_dataset(
    dataset_dir: str = 'repositories',
    skip_format: bool = False,
    max_file_size: int = 1 << 20,
    aux_fname: str = '_clean_dataset_aux',
):
    # assert dataset dir exists
    dataset_path = Path(dataset_dir)
    assert dataset_path.is_dir()

    # load or create pickled set to skip already cleaned source files
    # useful if this script is stopped while still doing work
    cache_path = Path(aux_fname)
    cleaned_set = set()
    if cache_path.is_file():
        with cache_path.open('r', encoding='utf-8') as fd:
            for line in fd:
                cleaned_set.add(line.strip())

    # count total files in the dataset directory
    total = len(list(dataset_path.rglob('*.py')))
    print(f'There are {total} source files in the dataset directory')

    # start formatting with tqdm progress
    pbar = tqdm(dataset_path.rglob('*.py'), total=total)
    for f in pbar:
        pbar.set_description(str(f))

        # skip already cleaned files
        if str(f) in cleaned_set:
            continue

        # skip directories that are named with suffix ".py"
        if not f.is_file():
            continue

        # skip source files with size greater than 1MB
        if f.stat().st_size > max_file_size:
            pbar.write(f'File too big: {f}')
            continue

        # read file contents into memory
        with f.open('r', encoding='utf-8') as fd:
            try:
                src = fd.read().strip()
                raise_error_if_empty(src)
                ast.parse(src)

                if not skip_format:
                    FormatFile(str(f))
            except (UnicodeDecodeError, ValueError, SyntaxError, lib2to3.pgen2.parse.ParseError) as e:
                pbar.write(f'{type(e).__name__}: {f}')
                f.unlink()
            else:
                with cache_path.open('a', encoding='utf-8') as fd:
                    fd.write(f'{str(f)}\n')

    # report cleaning stats
    delta = total - len(list(dataset_path.rglob('*.py')))
    print(f'Removed {delta} malformed files.')


def raise_error_if_empty(src: str):
    if src == '':
        raise ValueError('empty src')


if __name__ == '__main__':
    fire.Fire(clean_dataset)
