from preprocessing import collate_vocab_from_dir, tokenize

import codecs
import json
import os

SPECIAL_TOKENS = {
    '|Token.Text|_<|endofline|>': '\n',
    '|Token.Text|_<|space|>': ' ',
    '|Token.Text|_<|tab|>': '    ',
}


class Encoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

    def decode(self, tokens):
        src = ''

        for code in tokens:
            token = self.decoder[code]

            try:
                src += SPECIAL_TOKENS[token]
            except KeyError:
                token = token.split('_', 1)[1]
                src += token

        return src

    def encode(self, src):
        tokens = []

        for t_type, raw_token in tokenize(src):
            token = '|{}|_{}'.format(t_type, raw_token)

            try:
                tokens.append(self.encoder[token])
            except KeyError:
                token = '|{}|_<|unknown|>'.format(t_type)
                tokens.append(self.encoder[token])

        return tokens


def get_encoder():
    # attempt to load encoder from the saved json file
    if os.path.isfile('models/encoder.json'):
        with codecs.open('models/encoder.json', 'r') as f:
            encoder = json.load(f)
            return Encoder(encoder=encoder)

    # create a new vocabulary from the dataset
    vocabulary = collate_vocab_from_dir('repositories', output_data_file=True)
    encoder = {word: i for i, word in enumerate(vocabulary)}

    # persist created dictionary
    with codecs.open('models/encoder.json', 'w') as f:
        json.dump(encoder, f)

    return Encoder(encoder=encoder)
