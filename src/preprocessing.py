from collections import Counter
from pygments.lexers import Python3Lexer
from pygments.token import Token

import codecs
import json
import os


LEXER = Python3Lexer()
EXCLUDED_TOKEN_TYPES = set([
    Token.Comment.Hashbang,
    Token.Comment.Single,
    Token.Literal.String.Doc,
])
UNKNOWN_TOKEN_TYPES = set([
    Token.Name,
    Token.Name.Class,
    Token.Name.Function,
    Token.Name.Decorator,
])


def collate_vocab_from_dir(dirname, output_data_file=False):
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
        with codecs.open('vocab_data.txt', 'w', 'utf-8') as f:
            for token, count in counter.most_common():
                print('%s: %d' % (token, count), file=f)

    # create different unknown tokens for different program tokens (e.g., class/variable/func names)
    unknown_tokens = ['|{}|_<|unknown|>'.format(
        t_type) for t_type in UNKNOWN_TOKEN_TYPES]

    # delete values with count <= 10
    return unknown_tokens + [k for k, v in counter.items() if v > 10]


def tokenize(src):
    tokens = []
    single_line_string_flag = False
    multi_line_string_flag = False
    acc = ''

    for t_type, token in LEXER.get_tokens(src):
        # skip tokens that are just comments
        if t_type in EXCLUDED_TOKEN_TYPES:
            continue

        # react to start/end of multiline strings
        if token == "'''" or token == '"""':
            multi_line_string_flag = not multi_line_string_flag
            if not multi_line_string_flag:
                tokens.append('|Token.Literal.String.Single|_' + acc)
                acc = ''
            continue

        # don't treat strings inside multiline strings as tokens
        # as they may be documentation
        if multi_line_string_flag:
            acc += token
            continue

        # compress strings into one token
        # control start/end of string delimiters
        if t_type == Token.Literal.String.Single and token == "'":
            single_line_string_flag = not single_line_string_flag
            if not single_line_string_flag:
                tokens.append('|Token.Literal.String.Single|_' + acc)
                acc = ''
            continue

        # compress strings into one token
        # accumulate strings in between
        if t_type == Token.Literal.String.Single and single_line_string_flag:
            acc += token
            continue

        # treat multiple tabs as multiple different tokens
        # don't treat uneven spaces as a unique token since they are just
        # there to make spacing look good
        if token.isspace():
            if token == '\n':
                tokens.append('|Token.Text|_<|endofline|>')
                continue
            if token == ' ':
                tokens.append('|Token.Text|_<|space|>')
                continue

            if len(token) % 4 == 0:
                tokens.extend(['|Token.Text|_<|tab|>'] * (len(token) // 4))
                continue
            # treat spaces not divisible by 4 as spaces
            # since they are just there to align with the function call on top
            else:
                tokens.append('|Token.Text|_<|space|>')
                continue

        token = '|{}|_{}'.format(t_type, token)
        tokens.append(token)

    return tokens


def create_word2dev_dataset(dirname, wsize=2):
    assert os.path.isdir(dirname)
    data = []

    # attempt to load encoder from the saved json file
    if os.path.isfile('models/encoder.json'):
        with codecs.open('models/encoder.json', 'r') as f:
            vocabulary = json.load(f).keys()

    for root, _, files in os.walk(dirname):
        for pf in files:
            # skip files that are not python src codes
            if not pf.endswith('.py'):
                continue

            # open each src file and collate all unique tokens
            with codecs.open(os.path.join(root, pf), 'r', 'utf-8') as fd:
                for line in fd:
                    tokens = [token for token in tokenize(
                        line) if token in vocabulary]
                    data.append(tokens)

    return data
