from ccompletion.download_utils import get_repo_list, collate_python_files
from typing import Optional

import fire
import sys


def download(
    repositories: str = 'repositories.csv',
    access_token: Optional[str] = None,
    threshold: int = 1000
):
    repo_list = get_repo_list(
        name=repositories,
        star_count_threshold=threshold,
    )

    for reponame in repo_list:
        user, name = reponame.split('/', 1)

        try:
            collate_python_files(user, name, access_token=access_token)
        except (OSError, AssertionError) as e:
            print(f'WARN - Could not unpack {reponame}')
            print(e)


if __name__ == '__main__':
    fire.Fire(download)
