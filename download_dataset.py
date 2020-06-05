from ccompletion.dataset import get_repo_list, collate_python_files
from tqdm import tqdm

import fire
import os
import shutil


def download(repo_file='repository_list.txt', access_token=None):
    repo_list = get_repo_list(name=repo_file)
    pbar = tqdm(repo_list)

    for user, name in pbar:
        try:
            collate_python_files(user, name, access_token=access_token)
        except OSError as e:
            pbar.write('WARN - Could not unpack {}/{}'.format(user, name))
            pbar.write(str(e))

    # delete temporary directory
    if os.path.isdir('tmp'):
        shutil.rmtree('tmp')

if __name__ == '__main__':
    fire.Fire(download)
