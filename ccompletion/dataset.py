from pathlib import Path
from tqdm import tqdm

import codecs
import csv
import os
import re
import requests
import shutil
import tensorflow as tf


API_BASE_URL = 'https://api.github.com'
REPO_INFO_API = '/repos/{}/{}'
REPO_ZIP_URL = 'https://github.com/{}/{}/archive/{}.zip'
EXCLUDED_PATTERN = re.compile(r'(?:__init__\.py)|(?:test.+\.py)')


def get_latest_release_url(user, name, access_token=None):
    """
    Retrieves the download url of the repository as a compressed zip file.
    The downloaded zip file will contain the latest code of the
    default branch of the repository. Only repositories in Github are supported.

    Parameters:
    user (string): the username of the owner of the github repository
    name (string): the name of the target repository

    Returns:
    url (string): the download url of the repository as zip
    """
    latest_release_api = REPO_INFO_API.format(user, name)
    api_url = API_BASE_URL + latest_release_api
    headers = get_header_for_auth(access_token=access_token)
    r = requests.get(api_url, headers=headers)

    # Ensure API requests return OK
    if r.status_code != 200:
        raise AssertionError(r.text)

    # get repo's default branch
    default_branch = r.json()['default_branch']
    url = REPO_ZIP_URL.format(user, name, default_branch)

    return url, default_branch


def get_header_for_auth(access_token=None):
    if access_token is None:
        try:
            access_token = os.environ['GITHUB_ACCESS_TOKEN']
        except KeyError:
            err_msg = 'ERROR - "GITHUB_ACCESS_TOKEN" variable not found!'
            raise EnvironmentError(err_msg)

    auth_value = 'token {}'.format(access_token)
    return {'Authorization': auth_value}


def get_repo_list(name='repositories.csv', star_count_threshold=1000):
    # ensure csv file exists
    if not os.path.isfile(name):
        raise FileNotFoundError(f'"{name}" does not exist.')

    # the resulting repository list container
    repositories = []

    # csv file should be opened using UTF8 encoding
    # since repository names may contain non-ASCII letters
    with codecs.open(name, 'r', 'utf8', errors='replace') as fd:
        reader = csv.reader(fd, delimiter=',')

        # skip first line of csv file since it only contains headers
        next(reader)

        for repo_name, repo_stars in reader:
            if int(repo_stars) >= star_count_threshold:
                repositories.append(repo_name)
            else:
                # star count is already below the threshold
                # that is, all of the subsequent repositories
                # will have a star count below the threshold
                break

    return repositories


def collate_python_files(user, name, access_token=None):
    # ensure "repositories" directory exists
    if not os.path.isdir('repositories'):
        os.mkdir('repositories')

    # ensure parent directory for the user exists
    user_dir = os.path.join('repositories', user)
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)

    # skip if repository is already downloaded and extracted
    container_dir = os.path.join('repositories', user, name)
    if os.path.isdir(container_dir) and len(os.listdir(container_dir)) != 0:
        return

    # download repo's default branch and preserve python files
    url, _ = get_latest_release_url(user, name, access_token=access_token)
    filename = f'{user}_{name}.zip'
    path = tf.keras.utils.get_file(
        filename,
        origin=url,
        extract=True,
        cache_dir='.keras',
        cache_subdir='code-completion-repos',
    )

    # determine extracted directory name
    cache_dir_path = os.path.dirname(path)
    _dirs = [
        f
        for f in os.listdir(cache_dir_path)
        if os.path.isdir(os.path.join(cache_dir_path, f))
    ]
    assert len(_dirs) == 1, "More directories were found in extracted dir"
    extracted_dir_path = os.path.join(
        cache_dir_path,
        _dirs[0],
    )

    # filter out python source files and move it to 'repositories' directory
    extract_python_src_files(
        user,
        name,
        extracted_dir_path,
        exclude_tests=False,
    )

    # remove downloaded zip
    os.remove(path)
    print()


def extract_python_src_files(user, name, path, exclude_tests=True):
    # ensure extracted zip directory exists
    assert os.path.isdir(path), f"{path} directory does not exist!"

    # ensure target source files container directory exists
    container_directory = os.path.join('repositories', user, name)
    if not os.path.isdir(container_directory):
        os.mkdir(container_directory)

    # move all candidate python source files to the target container directory
    for root, dirs, files in os.walk(path):
        # remove all test directories
        if exclude_tests:
            dirs[:] = [d for d in dirs if not d.lower().startswith('test')]

        for f in files:
            # filter out non-python source files
            if not f.endswith('.py'):
                continue

            # include all source files if exclude_tests is False
            # filter out __init__.py or test*.py if exclude_tests is True
            if not exclude_tests or (exclude_tests and not re.match(EXCLUDED_PATTERN, f)):
                file_src_path = os.path.join(root, f)
                file_dst_path = os.path.join(container_directory, f)
                shutil.move(file_src_path, file_dst_path)

    # delete original project directory
    shutil.rmtree(path)
