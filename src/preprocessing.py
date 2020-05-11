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


def collate_vocab_from_dir(dirname):
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

    # delete values with count <= 10
    # and include unknown token
    return ['<|unknown|>'] + [k for k, v in counter.items() if v > 10]


def tokenize(src):
    tokens = []

    multi_line_string_flag = False
    for t_type, token in LEXER.get_tokens(src):
        # skip tokens that are just comments
        if t_type in EXCLUDED_TOKEN_TYPES:
            continue

        # don't treat strings inside multiline strings as tokens
        # as they may be documentation
        if multi_line_string_flag:
            continue

        # react to start/end of multiline strings
        if token == "'''" or token == '"""':
            multi_line_string_flag = not multi_line_string_flag
            continue

        # treat multiple tabs as multiple different tokens
        # don't treat uneven spaces as a unique token since they are just
        # there to make spacing look good
        if token.isspace():
            if len(token) % 4 == 0:
                tokens.append(' ' * 4)
            continue

        # treat this token as a token itself
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
