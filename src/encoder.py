from preprocessing import collate_vocab_from_dir, tokenize

import codecs
import json
import os


class Encoder:
    def __init__(self, encoder):
        self.encoder = encoder

    def encode(self, src):
        unknown = self.encoder['<|unknown|>']
        return [self.encoder.get(token, unknown) for token in tokenize(src)]


def get_encoder():
    # attempt to load encoder from the saved json file
    if os.path.isfile('models/encoder.json'):
        with codecs.open('models/encoder.json', 'r') as f:
            encoder = json.load(f)
            return Encoder(encoder=encoder)

    # create a new vocabulary from the dataset
    vocabulary = collate_vocab_from_dir('repositories')
    encoder = {word: i for i, word in enumerate(vocabulary)}

    # persist created dictionary
    with codecs.open('models/encoder.json', 'w') as f:
        json.dump(encoder, f)

    return Encoder(encoder=encoder)
