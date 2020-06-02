from ccompletion.encoder import get_encoder
from ccompletion.layers.attention import create_masks
from ccompletion.model import CC
from ccompletion.preprocessing import get_hparams

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

    print('=' * 80)
    print('INPUT: ', src)
    print('=' * 80)

    # if run on an implementation of tensorflow-gpu
    # training fails without the stuff below, idk why
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(
            physical_devices[0],
            enable=True,
        )

    # retrieve hyperparameters and encoder
    hparams = get_hparams()
    encoder = get_encoder()

    # create model and configure training checkpoint manager
    model = CC(**hparams)
    ckpt_path = './checkpoints/train'
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, 1)

    # check for latest checkpoint existence
    if not ckpt_manager.latest_checkpoint:
        print('ERROR - No checkpoints found.')
        print('ERROR - Train the model first before executing this script.')
        exit(1)

    # load model weights using the latest checkpoint
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

    # encode input source code
    input_eval = encoder.encode(src, add_start_token=True)
    input_eval = tf.expand_dims(input_eval, 0)

    # flag to determine when to stop predicting process
    should_stop = False

    # let model to output token until we get <|endofline|>
    while not should_stop:
        mask = create_masks(input_eval)
        predictions = model(input_eval, False, mask)
        predictions = predictions[:, -1, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        predicted_token = encoder.decode([predicted_id.numpy().item()])

        # concatenate predicted to source
        src += predicted_token

        # stop if we encounter end of line
        if predicted_token == '\n':
            should_stop = True
            continue

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.concat([input_eval, predicted_id], axis=-1)

    print('======================== OUTPUT ========================')
    print(src)
    print('======================== OUTPUT ========================')


if __name__ == '__main__':
    fire.Fire(interact)
