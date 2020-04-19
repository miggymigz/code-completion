from .github import download_latest_release

import codecs
import os
import re
import shutil

# regex pattern for python src filename filtering
EXCLUDED_PATTERN = re.compile(r'(?:__init__\.py)|(?:test_.+\.py)')


def get_repo_list():
    repo_list_filename = 'preprocessing/repository_list.txt'
    with codecs.open(repo_list_filename, 'r', 'utf-8') as f:
        for line in f:
            cleaned = line.strip()

            if cleaned:
                try:
                    user, name = cleaned.split('/')
                    yield user, name
                except ValueError:
                    print(
                        'INFO - "%s" was skipped because of invalid format.' % cleaned)
                    continue


def collate_python_files(user, name):
    # ensure "repositories" directory exists
    if not os.path.isdir('repositories'):
        os.mkdir('repositories')

    # ensure parent directory for the user exists
    user_dir = os.path.join('repositories', user)
    if not os.path.isdir(user_dir):
        os.mkdir(user_dir)

    # skip if repository is already downloaded and extracted
    container_dir = os.path.join('repositories', user, name)
    if os.path.isdir(container_dir):
        return

    # create pathname for the to-be-downloaded tarball
    output_path = 'repositories/{}/{}.tar.gz'.format(user, name)
    download_latest_release(user, name, output_path)
    extract_python_src_files(user, name, output_path)


def extract_python_src_files(user, repo_name, tarball_path):
    # ensure path exists and ends with ".tar.gz"
    assert os.path.isfile(tarball_path)
    assert tarball_path.endswith('.tar.gz')

    # unpack and delete tarball
    unpack_destination = os.path.join('repositories', user)
    shutil.unpack_archive(tarball_path, unpack_destination)
    os.unlink(tarball_path)

    # ensure target source files container directory exists
    container_directory = os.path.join(unpack_destination, repo_name)
    if not os.path.isdir(container_directory):
        os.mkdir(container_directory)

    # the name of the folder that contains the project code
    # is the repository name plus the version/tag
    project_path = [os.path.join(unpack_destination, f) for f in os.listdir(unpack_destination)
                    if os.path.isdir(os.path.join(unpack_destination, f))][0]

    # move all candidate python source files to the target container directory
    for root, _, files in os.walk(project_path):
        for f in files:
            if f.endswith('.py') and not re.match(EXCLUDED_PATTERN, f):
                file_src_path = os.path.join(root, f)
                file_dst_path = os.path.join(container_directory, f)
                shutil.move(file_src_path, file_dst_path)

    # delete original project directory
    shutil.rmtree(project_path)
