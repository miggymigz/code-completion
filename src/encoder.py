from preprocessing import collate_vocab_from_dir, tokenize, START_TOKEN

import codecs
import json
import os

SPECIAL_TOKENS = {
    START_TOKEN: '',
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

    def encode(self, src, add_start_token=False):
        tokens = []

        # add_start_token is a flag for convenience to automatically add the start token
        # at the start of each src tokens
        if add_start_token:
            tokens.append(self.encoder[START_TOKEN])

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

    # ensure models directory exists
    if not os.path.isdir('models'):
        os.mkdir('models')

    # persist created dictionary
    with codecs.open('models/encoder.json', 'w') as f:
        json.dump(encoder, f)

    # change hyperparameter n_vocab whenever encoder.json changes
    with codecs.open('models/hparams.json', 'r+', 'utf-8') as f:
        hparams = json.load(f)
        hparams['n_vocab'] = len(vocabulary)
        json.dump(hparams, f)

    return Encoder(encoder=encoder)
