from pathlib import Path
from tqdm import tqdm

import ast
import fire


def clean_dataset(dataset_dir: str = 'repositories'):
    # assert dataset dir exists
    dataset_path = Path(dataset_dir)
    assert dataset_path.is_dir()

    # count total files in the dataset directory
    total = len(list(dataset_path.rglob('*.py')))
    print(f'There are {total} source files in the dataset directory')

    # start formatting with tqdm progress
    pbar = tqdm(dataset_path.rglob('*.py'), total=total)
    for f in pbar:
        pbar.set_description(str(f))

        # skip directories that are named with suffix ".py"
        if not f.is_file():
            continue

        # read file contents into memory
        with f.open('r', encoding='utf-8') as fd:
            try:
                src = fd.read().strip()
                raise_error_if_empty(src)
                ast.parse(src)
            except (UnicodeDecodeError, ValueError, SyntaxError) as e:
                pbar.write(f'{type(e).__name__}: {f}')
                f.unlink()

    # report cleaning stats
    delta = total - len(list(dataset_path.rglob('*.py')))
    print(f'Removed {delta} malformed files.')


def raise_error_if_empty(src: str):
    if src == '':
        raise ValueError('empty src')


if __name__ == '__main__':
    fire.Fire(clean_dataset)
