from collections import Counter
from pygments.lexers import Python3Lexer
from pygments.token import Token

import tensorflow as tf
import codecs
import json
import numpy as np
import os


LEXER = Python3Lexer()
EXCLUDED_TOKEN_TYPES = set([
    Token.Comment.Hashbang,
    Token.Comment.Single,
    Token.Literal.String.Doc,
])
UNKNOWN_TOKEN_TYPES = set([
    Token.Literal.Number.Float,
    Token.Literal.Number.Hex,
    Token.Literal.Number.Integer,
    Token.Literal.Number.Oct,
    Token.Literal.String.Double,
    Token.Literal.String.Escape,
    Token.Literal.String.Interpol,
    Token.Literal.String.Single,
    Token.Name,
    Token.Name.Class,
    Token.Name.Decorator,
    Token.Name.Exception,
    Token.Name.Function,
    Token.Name.Namespace,
])
START_TOKEN = '<|start|>'
DEFAULT_HPARAMS = {
    'n_vocab': 0,
    'n_embd': 768,
    'n_ctx': 1024,
    'n_head': 12,
    'n_layer': 12,
}


def collate_vocab_from_dir(dirname, threshold=10, output_data_file=False):
    assert os.path.isdir(dirname)
    counter = Counter()

    for root, _, files in os.walk(dirname):
        for pf in files:
            # skip files that are not python src codes
            if not pf.endswith('.py'):
                continue

            # open each src file and collate all unique tokens
            with codecs.open(os.path.join(root, pf), 'r', 'utf-8') as fd:
                src_code = fd.read()
                tokens = tokenize(src_code)
                counter.update(tokens)

    if output_data_file:
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


def tokenize(src, trim_leading_newlines=True):
    tokens = []
    single_line_start = None
    multi_line_start = None
    acc = ''

    for t_type, token in LEXER.get_tokens(src):
        # skip tokens that are just comments
        if t_type in EXCLUDED_TOKEN_TYPES:
            continue

        # react to start/end of multiline strings
        if token == "'''" or token == '"""':
            if multi_line_start is None:
                multi_line_start = token
            # end token matches start token
            elif multi_line_start == token:
                tokens.append((t_type, acc))
                acc = ''
                multi_line_start = None
            else:
                acc += token
            continue

        # don't treat strings inside multiline strings as tokens
        # as they may be documentation
        if multi_line_start is not None:
            acc += token
            continue

        # compress strings into one token
        # control start/end of string delimiters
        if token == "'" or token == '"':
            if single_line_start is None:
                single_line_start = token
            elif single_line_start == token:
                tokens.append((t_type, acc))
                acc = ''
                single_line_start = None
            else:
                acc += token
            continue

        # compress strings into one token
        # accumulate strings in between
        if single_line_start is not None:
            acc += token
            continue

        # treat multiple tabs as multiple different tokens
        # don't treat uneven spaces as a unique token since they are just
        # there to make spacing look good
        if token.isspace():
            if token == '\n':
                tokens.append((t_type, '<|endofline|>'))
                continue
            if token == ' ':
                tokens.append((t_type, '<|space|>'))
                continue
            if len(token) % 4 == 0:
                single_tab = (t_type, '<|tab|>')
                tokens.extend([single_tab] * (len(token) // 4))
                continue
            # treat spaces not divisible by 4 as spaces
            # since they are just there to align with the function call on top
            else:
                tokens.append((t_type, '<|space|>'))
                continue

        # other tokens need not to be preprocessed
        tokens.append((t_type, token))

    if trim_leading_newlines:
        while tokens[0][1] == '<|endofline|>':
            tokens.pop(0)

    return tokens


def collate_training_dataset(name='dataset.npz', batch_size=64, buffer_size=10000, sequence_length=100):
    # ensure compressed dataset file exists
    if not os.path.isfile(name):
        print('ERROR - "dataset.npz" not found.')
        print('ERROR - Encode the repositories first using encode.py')
        exit(1)

    # load encoded tokens from the compressed dataset file
    with np.load(name, allow_pickle=True) as npz:
        # ensure "token_chunks" array exists in the file
        if "token_chunks" not in npz.files:
            print('ERROR - "dataset.npz" does not contain the token chunks.')
            print('ERROR - Be sure to encode the repositories by using encode.py')
            exit(1)

        # retrieve token chunks from the compressed dataset file
        for src_tokens in npz['token_chunks']:
            assert len(src_tokens) > 1
            yield (src_tokens[:-1], src_tokens[1:])


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


def get_hparams(name='models/hparams.json'):
    # return default hparams if hparams.json does not exist
    if not os.path.isfile(name):
        return DEFAULT_HPARAMS

    # read hparams.json and return its contents
    with codecs.open(name, 'r', 'utf-8') as fd:
        try:
            return json.load(fd)
        except json.JSONDecodeError:
            return DEFAULT_HPARAMS


def save_hparams(name='models/hparams.json', **kwargs):
    # load hparams from file if it exists
    if os.path.isfile(name):
        with codecs.open(name, 'r', 'utf-8') as fd:
            hparams = json.load(fd)
    # otherwise create new hparams from default hparams
    else:
        hparams = DEFAULT_HPARAMS

    # update hparams from keyword arguments
    hparams.update(kwargs)

    # persist updated hparams to file
    with codecs.open(name, 'w', 'utf-8') as fd:
        json.dump(hparams, fd, indent=4)
