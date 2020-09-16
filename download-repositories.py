from ccompletion.download_utils import get_repo_list, collate_python_files
from pathlib import Path
from tqdm import tqdm
from typing import Optional

import fire
import sys


def download(
    repositories: str = 'repositories.csv',
    output_dir: str = 'repositories',
    access_token: Optional[str] = None,
    threshold: int = 500,
):
    repos_csv_path = Path(repositories)
    repos_output_path = Path(output_dir)

    # get the list of repositories to be downloaded
    repo_list = get_repo_list(
        name=repos_csv_path,
        star_count_threshold=threshold,
    )

    for reponame in tqdm(repo_list):
        try:
            collate_python_files(
                reponame,
                output_path=repos_output_path,
                access_token=access_token,
            )
        except (OSError, AssertionError) as e:
            print(f'WARN - Could not unpack {reponame}')
            print(e)


if __name__ == '__main__':
    fire.Fire(download)
