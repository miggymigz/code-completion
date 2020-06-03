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
        try:
            while tokens[0][1] == '<|endofline|>':
                tokens.pop(0)
        except IndexError:
            pass

    return tokens
