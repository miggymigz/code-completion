from encoder import get_encoder
from model import build_model

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

    print('INPUT: ', src)

    # retrieve model hyperparameters
    with open(os.path.join('models', 'hparams.json')) as f:
        hparams = json.load(f)
        n_vocab = hparams['n_vocab']
        n_embd = hparams['n_embd']

    # retrieve encoder
    encoder = get_encoder()

    # because of the way the RNN state is passed from timestep to timestep,
    # the model only accepts a fixed batch size once built
    model = build_model(n_vocab=n_vocab, n_embd=n_embd, n_batch=1)

    # load pre-trained weights
    checkpoint_dir = 'training_checkpoints'
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    model.reset_states()
    model.summary()

    # encode input source code
    input_eval = [token for token in encoder.encode(src)]
    input_eval = tf.expand_dims(input_eval, 0)

    # flag to determine when to stop predicting process
    should_stop = False

    # let model to output token until we get <|endofline|>
    while not should_stop:
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        predicted_token = encoder.decode([predicted_id])

        # concatenate predicted to source
        src += predicted_token

        # stop if we encounter end of line
        if predicted_token == '\n':
            should_stop = True
            continue

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

    print('======================== OUTPUT ========================')
    print(src)
    print('======================== OUTPUT ========================')


if __name__ == '__main__':
    fire.Fire(interact)
