from preprocessing import collate_vocab_from_dir, tokenize, START_TOKEN, get_hparams, save_hparams

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

    print('INFO - encoder.json does not exist.')
    print('INFO - Will generate encoder.json from the downloaded repositories')

    # create a new vocabulary from the dataset
    vocabulary = collate_vocab_from_dir('repositories', output_data_file=True)
    encoder = {word: i for i, word in enumerate(vocabulary)}

    # ensure models directory exists
    if not os.path.isdir('models'):
        os.mkdir('models')

    # change hyperparameter n_vocab whenever encoder.json changes
    # read all other hyperparameters
    hparams = get_hparams(name='models/hparams.json')
    hparams['n_vocab'] = len(vocabulary)
    save_hparams(name='models/hparams.json', **hparams)

    # persist created dictionary
    with codecs.open('models/encoder.json', 'w', 'utf-8') as f:
        json.dump(encoder, f)

    return Encoder(encoder=encoder)
