from ccompletion.dataset import get_repo_list, collate_python_files

import fire
import sys


def download(repositories='repositories.csv', access_token=None, threshold=1000):
    repo_list = get_repo_list(
        name=repositories,
        star_count_threshold=threshold,
    )

    for reponame in repo_list:
        user, name = reponame.split('/', 1)

        try:
            collate_python_files(user, name, access_token=access_token)
        except:
            error = sys.exc_info()[0]
            print(f'WARN - Could not unpack {reponame}')
            print(error)


if __name__ == '__main__':
    fire.Fire(download)
