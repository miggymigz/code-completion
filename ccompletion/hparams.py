import os
import codecs
import json
import tensorflow as tf


DEFAULT_HPARAMS = {
    'n_vocab': 0,
    'n_embd': 768,
    'n_ctx': 1024,
    'n_head': 12,
    'n_layer': 2,
}


def get_hparams(name='models/hparams.json'):
    # return default hparams if hparams.json does not exist
    if not os.path.isfile(name):
        return DEFAULT_HPARAMS

    # read hparams.json and return its contents
    with codecs.open(name, 'r', 'utf-8') as fd:
        try:
            return json.load(fd)
        except json.JSONDecodeError:
            return DEFAULT_HPARAMS


def save_hparams(name='models/hparams.json', **kwargs):
    # load hparams from file if it exists
    if os.path.isfile(name):
        with codecs.open(name, 'r', 'utf-8') as fd:
            hparams = json.load(fd)
    # otherwise create new hparams from default hparams
    else:
        hparams = DEFAULT_HPARAMS

    # update hparams from keyword arguments
    hparams.update(kwargs)

    # persist updated hparams to file
    with codecs.open(name, 'w', 'utf-8') as fd:
        json.dump(hparams, fd, indent=4)


class LRCustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LRCustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
