from encoder import get_encoder
from model import build_model
from preprocessing import collate_training_dataset

import tensorflow as tf
import fire
import json
import os


def train_model(n_batch=64):
    # training fails without the stuff below, idk why
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # retrieve model hyperparameters
    with open(os.path.join('models', 'hparams.json')) as f:
        hparams = json.load(f)
        n_vocab = hparams['n_vocab']
        n_embd = hparams['n_embd']

    # retrieve encoder and training dataset
    encoder = get_encoder()
    dataset = collate_training_dataset(encoder)

    # create model based on hyperparameters
    model = build_model(n_vocab=n_vocab, n_embd=n_embd, n_batch=n_batch)
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    model.summary()

    # configure checkpoint
    checkpoint_dir = 'training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True,
        save_best_only=True,
        monitor='loss',
        verbose=1,
    )

    # start training
    model.fit(dataset, epochs=100, callbacks=[checkpoint_callback])


if __name__ == '__main__':
    fire.Fire(train_model)
