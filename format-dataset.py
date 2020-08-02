from ccompletion.utils import get_total_file_count
from subprocess import PIPE, run
from tqdm import tqdm

import fire
import os


def format_dataset(dataset_dir='repositories'):
    # assert dataset dir exists
    if not os.path.isdir(dataset_dir):
        print(f'Dataset directory "{dataset_dir}" does not exist')
        return

    # count total files in the dataset directory
    total = get_total_file_count(dataset_dir)
    print(f'There are {total} files in dataset directory')

    # start formatting with tqdm progress
    with tqdm(total=total) as t:
        for root, _, files in os.walk(dataset_dir):
            for pf in files:
                # skip files that are not python src codes
                if not pf.endswith('.py'):
                    t.update()
                    continue

                # get source file path
                pf_path = os.path.join(root, pf)
                t.set_description(pf_path)
                pf_path = os.path.abspath(pf_path)

                # invoke autopep8 using subprocess
                command = ['autopep8', '--in-place', pf_path]
                result = run(
                    command,
                    stdout=PIPE,
                    stderr=PIPE,
                    universal_newlines=True,
                )

                if result.stdout:
                    t.write(str(result.stdout))

                if result.stderr:
                    t.write(str(result.stderr))

                t.update()


if __name__ == '__main__':
    fire.Fire(format_dataset)
