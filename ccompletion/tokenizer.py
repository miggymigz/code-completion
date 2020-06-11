from transformers import PreTrainedTokenizer
from .dataset import collate_vocab_from_dir
from .hparams import get_hparams, save_hparams
from .preprocessing import tokenize, START_TOKEN


import codecs
import json
import os

SPECIAL_TOKENS = {
    START_TOKEN: '',
    '|Token.Text|_<|endofline|>': '\n',
    '|Token.Text|_<|space|>': ' ',
    '|Token.Text|_<|tab|>': '    ',
}


class CodeCompletionTokenizer(PreTrainedTokenizer):
    def __init__(self, encoder, unk_token='<|endofsrc|>', bos_token='<|endofsrc|>', eos_token='<|endofsrc|>', **kwargs):
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, code):
        tokens = tokenize(code)
        return tokens[:-1]

    def _convert_token_to_id(self, token):
        t_type, raw_token = token
        token = '|{}|_{}'.format(t_type, raw_token)

        try:
            return self.encoder[token]
        except KeyError:
            token = '|{}|_<|unknown|>'.format(t_type)
            return self.encoder[token]

    def _convert_id_to_token(self, index):
        return self.decoder[index]

    def convert_tokens_to_string(self, tokens):
        src = ''

        for t_type, raw_token in tokens:
            concatenated_token = '|{}|_{}'.format(t_type, raw_token)

            try:
                src += SPECIAL_TOKENS[concatenated_token]
            except KeyError:
                if 'String' in t_type:
                    src += "'{}'".format(raw_token)
                else:
                    src += raw_token

        return src


def get_tokenizer(threshold=20):
    # attempt to load encoder from the saved json file
    if os.path.isfile('models/encoder.json'):
        with codecs.open('models/encoder.json', 'r') as f:
            encoder = json.load(f)
            return CodeCompletionTokenizer(encoder=encoder)

    print('INFO - encoder.json does not exist.')
    print('INFO - Will generate encoder.json from the downloaded repositories')

    # create a new vocabulary from the dataset
    vocabulary = collate_vocab_from_dir(
        'repositories',
        threshold=threshold,
        output_data_file=True
    )
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

    return CodeCompletionTokenizer(encoder=encoder)
