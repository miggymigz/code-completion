from collections import Counter
from tqdm import tqdm

from .hparams import get_hparams
from .preprocessing import tokenize, START_TOKEN, UNKNOWN_TOKEN_TYPES

import codecs
import numpy as np
import os
import re
import requests
import shutil
import tensorflow as tf


API_BASE_URL = 'https://api.github.com'
REPO_INFO_API = '/repos/{}/{}'
REPO_ZIP_URL = 'https://github.com/{}/{}/archive/{}.zip'
EXCLUDED_PATTERN = re.compile(r'(?:__init__\.py)|(?:test_.+\.py)')


def get_latest_release_tarball_url(user, name, access_token=None):
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

    # return download link for the source zip
    return REPO_ZIP_URL.format(user, name, default_branch)


def download_latest_release(user, name, path, access_token=None):
    """
    Downloads the github repository at github.com/{user}/{name}
    as a zip. The downloaded zip will contain the latest source code
    of the default branch of the repository. The downloaded file will be placed in path.

    Parameters:
    user (string): the username of the owner of the github repository
    name (string): the name of the target repository 
    path (string): the path in which the downloaded zip file will be written to
    """
    url = get_latest_release_tarball_url(user, name, access_token=access_token)
    r = requests.get(url, stream=True)

    with open(path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)


def get_header_for_auth(access_token=None):
    if access_token is None:
        try:
            access_token = os.environ['GITHUB_ACCESS_TOKEN']
        except KeyError:
            err_msg = 'ERROR - "GITHUB_ACCESS_TOKEN" variable not found!'
            raise EnvironmentError(err_msg)

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


def collate_python_files(user, name, access_token=None):
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
    download_latest_release(user, name, output_path, access_token=access_token)
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


def download_repositories(name='repository_list.txt', access_token=None):
    repo_list = get_repo_list(name=name)
    for repo_user, repo_name in tqdm(repo_list):
        collate_python_files(repo_user, repo_name, access_token=access_token)

    # delete temporary directory
    if os.path.isdir('tmp'):
        shutil.rmtree('tmp')


def collate_vocab_from_dir(dirname, threshold=10, output_data_file=False):
    assert os.path.isdir(dirname)
    counter = Counter()

    # count total files for tqdm progress
    total = 0
    for _, _, files in os.walk(dirname):
        total += len(files)

    print('INFO - Tokenizing source files...')
    with tqdm(total=total) as t:
        for root, _, files in os.walk(dirname):
            for pf in files:
                # skip files that are not python src codes
                if not pf.endswith('.py'):
                    t.update()
                    continue

                # open each src file and collate all unique tokens
                pf_path = os.path.join(root, pf)
                with codecs.open(pf_path, 'r', 'utf8', errors='replace') as fd:
                    src_code = fd.read()
                    tokens = tokenize(src_code)
                    counter.update(tokens)
                    t.update()

    if output_data_file:
        print('INFO - Writing vocab statistics...')
        create_dataset_summary_file(counter, threshold=threshold)

    # create different unknown tokens for different program tokens (e.g., class/variable/func names)
    unknown_tokens = ['|{}|_<|unknown|>'.format(
        t_type) for t_type in UNKNOWN_TOKEN_TYPES]

    # delete values with count <= 10
    # but retain tokens of that is not a literal or a constant type
    filtered_tokens = []
    for (t_type, token), frequency in counter.items():
        if frequency > threshold or t_type not in UNKNOWN_TOKEN_TYPES:
            token = '|{}|_{}'.format(t_type, token)
            filtered_tokens.append(token)

    # add start token
    return [START_TOKEN] + unknown_tokens + filtered_tokens


def collate_training_dataset(name='dataset.npz'):
    # ensure compressed dataset file exists
    if not os.path.isfile(name):
        print('ERROR - "dataset.npz" not found.')
        print('ERROR - Encode the repositories first using encode.py')
        exit(1)

    # get the maximum token length from hparams
    hparams = get_hparams()
    max_seq_len = hparams['n_ctx']

    # load encoded tokens from the compressed dataset file
    with np.load(name, allow_pickle=True) as npz:
        # ensure "token_chunks" array exists in the file
        if "token_chunks" not in npz.files:
            print('ERROR - "dataset.npz" does not contain the token chunks.')
            print('ERROR - Be sure to encode the repositories by using encode.py')
            exit(1)

        # retrieve token chunks from the compressed dataset file
        for src_tokens in npz['token_chunks']:
            length = len(src_tokens)
            assert length > 1,\
                "ERROR - src tokens' length should be atleast greater than 1"

            # divide longer tokens into chunks if it exceeds max_seq_len
            if length > max_seq_len:
                chunks_length = int(np.ceil(length / max_seq_len))

                for j in range(chunks_length):
                    start_index = j * max_seq_len
                    end_index = (j+1) * max_seq_len
                    token_chunk = src_tokens[start_index: end_index]
                    yield token_chunk[:-1], token_chunk[1:]
            else:
                yield src_tokens[:-1], src_tokens[1:]


def create_dataset_summary_file(counter, threshold):
    with codecs.open('vocab_data.txt', 'w', 'utf-8') as f:
        rare_types = {}

        # print each token's frequency
        for (t_type, raw_token), count in counter.most_common():
            token = '|{}|_{}'.format(t_type, raw_token)
            print('%s: %d' % (token, count), file=f)

            # collect raw tokens of rare types
            if count <= threshold:
                if t_type in rare_types:
                    rare_types[t_type].append(repr(raw_token))
                else:
                    rare_types[t_type] = [repr(raw_token)]

        print('\n', file=f)

        # print out token types that didn't pass threshold
        # also print out the corresponding raw tokens of each type
        print('TOKENS BELOW WILL BE TREATED AS UNKNOWNS (with exceptions ofc)', file=f)
        for k, v in rare_types.items():
            print('{} ==> {}'.format(k, v), file=f)


def get_tf_dataset(*, batch_size, buffer_size):
    def tf_encode(x, y):
        x.set_shape([None])
        y.set_shape([None])

        return x, y

    dataset = tf.data.Dataset.from_generator(
        collate_training_dataset,
        output_types=(tf.int32, tf.int32),
    )
    dataset = dataset.map(tf_encode)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
