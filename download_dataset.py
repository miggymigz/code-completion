from tqdm import tqdm

import codecs
import os
import re
import requests
import shutil


API_BASE_URL = 'https://api.github.com'
REPO_INFO_API = '/repos/{}/{}'
REPO_ZIP_URL = 'https://github.com/{}/{}/archive/{}.zip'
EXCLUDED_PATTERN = re.compile(r'(?:__init__\.py)|(?:test_.+\.py)')


def get_latest_release_tarball_url(user, name):
    latest_release_api = REPO_INFO_API.format(user, name)
    api_url = API_BASE_URL + latest_release_api
    r = requests.get(api_url, headers=get_header_for_auth())

    # Ensure API requests return OK
    if r.status_code != 200:
        raise AssertionError(r.text)

    # get repo's default branch
    default_branch = r.json()['default_branch']

    # return download link for the source zip
    return REPO_ZIP_URL.format(user, name, default_branch)


def download_latest_release(user, name, path):
    url = get_latest_release_tarball_url(user, name)
    r = requests.get(url, stream=True)

    with open(path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


def get_header_for_auth():
    # retrieve access token from environment vars
    if 'GITHUB_ACCESS_TOKEN' not in os.environ:
        raise EnvironmentError(
            'Error: "GITHUB_ACCESS_TOKEN" variable not found!')

    access_token = os.environ['GITHUB_ACCESS_TOKEN']
    auth_value = 'token {}'.format(access_token)

    return {'Authorization': auth_value}


def get_repo_list(name='repository_list.txt'):
    repos = []

    with codecs.open(name, 'r', 'utf-8') as f:
        for line in f:
            cleaned = line.strip()

            # skip lines with that starts with #
            if line.startswith('#'):
                continue

            if cleaned:
                try:
                    user, name = cleaned.split('/')
                    repos.append((user, name))
                except ValueError:
                    print(
                        'INFO - "%s" was skipped because of invalid format.' % cleaned)
                    continue

    return repos


def collate_python_files(user, name):
    # ensure "repositories" directory exists
    if not os.path.isdir('repositories'):
        os.mkdir('repositories')

    # ensure parent directory for the user exists
    user_dir = os.path.join('repositories', user)
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)

    # ensure temporary directory exists
    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    # skip if repository is already downloaded and extracted
    container_dir = os.path.join('repositories', user, name)
    if os.path.isdir(container_dir):
        return

    # create pathname for the to-be-downloaded tarball
    output_path = 'tmp/{}_{}.zip'.format(user, name)
    download_latest_release(user, name, output_path)
    extract_python_src_files(user, name, output_path)


def extract_python_src_files(user, repo_name, tarball_path):
    # ensure path exists and ends with ".zip"
    assert os.path.isfile(tarball_path)
    assert tarball_path.endswith('.zip')

    # unpack and delete tarball
    unpack_destination = os.path.join('tmp', user + '_' + repo_name)
    shutil.unpack_archive(tarball_path, unpack_destination)

    # ensure target source files container directory exists
    container_directory = os.path.join('repositories', user, repo_name)
    if not os.path.isdir(container_directory):
        os.mkdir(container_directory)

    # the name of the folder that contains the project code
    # is the repository name plus the version/tag
    project_path = [os.path.join(unpack_destination, f) for f in os.listdir(unpack_destination)
                    if os.path.isdir(os.path.join(unpack_destination, f))][0]

    # move all candidate python source files to the target container directory
    for root, dirs, files in os.walk(project_path):
        # remove all test directories
        for d in dirs:
            if d.startswith('test'):
                shutil.rmtree(os.path.join(root, d))

        for f in files:
            if f.endswith('.py') and not re.match(EXCLUDED_PATTERN, f):
                file_src_path = os.path.join(root, f)
                file_dst_path = os.path.join(container_directory, f)
                shutil.move(file_src_path, file_dst_path)

    # delete original project directory
    shutil.rmtree(project_path)


if __name__ == '__main__':
    repo_list = get_repo_list()
    for repo_user, repo_name in tqdm(repo_list):
        collate_python_files(repo_user, repo_name)

    # delete temporary directory
    if os.path.isdir('tmp'):
        shutil.rmtree('tmp')

    print('INFO - Download dataset task - FINISHED')
