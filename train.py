from ccompletion.dataset import get_tf_dataset
from ccompletion.hparams import LRCustomSchedule, get_hparams
from ccompletion.layers.attention import create_masks
from ccompletion.model import CC

import tensorflow as tf
import json
import os
import time
import fire


def train(epochs=1000, buffer_size=10000, batch_size=8, max_to_keep=5):
    dataset = get_tf_dataset(buffer_size=buffer_size, batch_size=batch_size)
    hparams = get_hparams()
    print(hparams)

    model = CC(**hparams)
    model.create_optimizer()
    model.create_checkpoint_manager(
        checkpoint_path='checkpoints/train',
        max_to_keep=max_to_keep,
        load_model=True,
    )
    model.fit(dataset, epochs=epochs)


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
    else:
        print('WARNING - No GPU was detected.')
        print('WARNING - Training will take a very long time.')

    fire.Fire(train)
