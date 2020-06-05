from ccompletion.dataset import get_repo_list, collate_python_files
from tqdm import tqdm

import fire
import os
import shutil


def download(repo_file='repository_list.txt', access_token=None):
    repo_list = get_repo_list(name=repo_file)
    for repo_user, repo_name in tqdm(repo_list):
        collate_python_files(repo_user, repo_name, access_token=access_token)

    # delete temporary directory
    if os.path.isdir('tmp'):
        shutil.rmtree('tmp')

if __name__ == '__main__':
    fire.Fire(download)
