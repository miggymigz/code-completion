from tqdm import tqdm

from ccompletion.dataset import download_repositories

if __name__ == '__main__':
    repo_file = 'repository_list.txt'
    download_repositories(name=repo_file)
    print('INFO - Download dataset task - FINISHED')
