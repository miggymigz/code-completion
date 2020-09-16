from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ccompletion.download_utils import get_repo_list, collate_python_files

import fire
import sys
import warnings


def download(
    repositories: str = 'repositories.csv',
    output_dir: str = 'repositories',
    access_token: Optional[str] = None,
    threshold: int = 500,
):
    # convert path strings to Path
    repos_csv_path = Path(repositories)
    repos_output_path = Path(output_dir)

    # get the list of repositories to be downloaded
    repo_list = get_repo_list(
        name=repos_csv_path,
        star_count_threshold=threshold,
    )

    # download repositories locally
    for reponame in tqdm(repo_list):
        try:
            collate_python_files(
                reponame,
                output_path=repos_output_path,
                access_token=access_token,
            )
        except FileNotFoundError:
            # for some reason, the repository could not be found anymore
            # maybe it was set to private or deleted entirely
            warnings.warn(f'{reponame} is either private or deleted.')


if __name__ == '__main__':
    fire.Fire(download)
