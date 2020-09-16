from pathlib import Path
from tqdm import tqdm
from typing import Optional

import codecs
import csv
import os
import re
import requests
import shutil
import tensorflow as tf
import uuid
import zipfile


API_BASE_URL = 'https://api.github.com'
REPO_INFO_API = '/repos/{}/{}'
REPO_ZIP_URL = 'https://github.com/{}/{}/archive/{}.zip'


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


def get_repo_list(name: Path, star_count_threshold: int):
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


def collate_python_files(reponame: str, output_path: Path, access_token: Optional[str]):
    # split repo name into its username and repository name
    # e.g., donnemartin/system-design-primer
    user, name = reponame.split('/', 1)

    # ensure output directory exists
    output_path.mkdir(parents=False, exist_ok=True)

    # ensure parent directory for the user exists
    user_path = output_path / user
    user_path.mkdir(parents=False, exist_ok=True)

    # skip if repository is already downloaded and extracted
    # empty directories are okay because there are repositories
    # that does not contain any python script (or were filtered as unwanted)
    container_path = user_path / name
    if container_path.exists():
        return

    # download repo's default branch and preserve python files
    url, _ = get_latest_release_url(user, name, access_token=access_token)
    filename = f'{user}_{name}.zip'
    path = Path(tf.keras.utils.get_file(
        filename,
        origin=url,
        extract=True,
        cache_dir='.keras',
        cache_subdir='code-completion-repos',
    ))

    # determine extracted directory name
    top_dir = get_top_dir(path)
    extracted_dir_path = path.parent / top_dir

    # filter out python source files and move it to 'repositories' directory
    extract_python_src_files(
        user,
        name,
        extracted_dir_path,
        output_path=output_path,
    )

    # remove downloaded zip
    path.unlink()


def extract_python_src_files(user: str, name: str, path: Path, output_path: Path):
    # ensure extracted zip directory exists
    assert path.exists(), f"{path} directory does not exist!"

    # ensure target source files container directory exists
    container_path = output_path / user / name
    container_path.mkdir(parents=False, exist_ok=False)

    # move all candidate python source files to the target container directory
    for f in path.glob('**/*'):
        # only retain Python source files
        if str(f).endswith('.py'):
            try:
                dst_fname = f.name
                shutil.move(f, container_path / dst_fname)
            except FileExistsError:
                dst_fname = str(uuid.uuid1()) + '.py'
                shutil.move(f, container_path / dst_fname)

    # delete original project directory
    shutil.rmtree(path)


def get_top_dir(repo_path: Path):
    """
    Repository archives usually contains a single directory which contains
    the actual files of that repository. When downloading the file, 
    `tf.keras.utils.get_file` will automatically extract this directory. So,
    this function returns the name of that directory so that it can be used
    to filter the files further.
    """
    zf = zipfile.ZipFile(repo_path)
    topdir = list(set([
        y
        for x in zf.namelist()
        if (y := os.path.dirname(x)) and '/' not in y and '\\' not in y
    ]))

    # only one directory name should remain
    assert len(topdir) == 1, f"{repo_path}'s format is unknown"

    return topdir[0]
