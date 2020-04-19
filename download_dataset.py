from preprocessing.repo_downloader import get_repo_list, collate_python_files
from tqdm import tqdm

if __name__ == '__main__':
    repo_list = get_repo_list()
    for repo_user, repo_name in tqdm(repo_list):
        collate_python_files(repo_user, repo_name)
    print('INFO - Download dataset task - FINISHED')
