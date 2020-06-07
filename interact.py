from ccompletion.encoder import get_encoder
from ccompletion.layers.attention import create_masks
from ccompletion.model import CC
from ccompletion.hparams import get_hparams

import tensorflow as tf
import codecs
import fire
import json
import os


def interact(src=None, file=None):
    # attempt to read file if src is not given
    if src is None:
        assert file is not None
        assert os.path.isfile(file)

        with codecs.open(file, 'r', 'utf-8') as f:
            src = f.read()
    # treat src as the actual source code string
    else:
        assert file is None

    print('=' * 36 + ' INPUT ' + '=' * 37)
    print(src)
    print('=' * 36 + ' INPUT ' + '=' * 37)

    # retrieve hyperparameters and encoder
    hparams = get_hparams()
    encoder = get_encoder()

    # create model and configure training checkpoint manager
    model = CC(**hparams)
    model.load_checkpoint('./checkpoints/train')

    # encode input source code
    input_eval = encoder.encode(src, add_start_token=True)
    input_eval = tf.expand_dims(input_eval, 0)

    # flag to determine when to stop predicting process
    should_stop = False
    n_generated_samples = 0

    # let model to output token until we get <|endofline|>
    while not should_stop:
        mask = create_masks(input_eval)
        predictions = model(input_eval, False, mask)
        predictions = predictions[:, -1:, :]

        top_k = tf.math.top_k(predictions, k=10)
        probable_token_ids = tf.squeeze(top_k[1]).numpy()
        predicted_id_idx, predicted_token = choose_token(
            encoder, probable_token_ids)
        predicted_id = top_k[1][:, :, predicted_id_idx]

        # concatenate predicted to source
        src += predicted_token
        n_generated_samples += 1

        # stop if we encounter end of line
        if predicted_token == '\n' and n_generated_samples > 1:
            should_stop = True
            continue

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.concat([input_eval, predicted_id], axis=-1)

    print('=' * 36 + ' OUTPUT ' + '=' * 36)
    print(src)
    print('=' * 36 + ' OUTPUT ' + '=' * 36)


def choose_token(encoder, token_ids):
    for i, token_id in enumerate(token_ids):
        decoded_token = encoder.decode([token_id])

        if decoded_token == '<|unknown|>':
            try:
                probable_token_id = token_ids[i+1]
                probable_token = encoder.decode([probable_token_id])
                return i+1, probable_token
            except IndexError:
                return i, '<|unknown|>'
        else:
            return i, decoded_token


if __name__ == '__main__':
    # if run on an implementation of tensorflow-gpu
    # training fails without the stuff below, idk why
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(
                device,
                enable=True,
            )

    fire.Fire(interact)
