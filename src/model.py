from encoder import get_encoder
from preprocessing import collate_training_dataset

import tensorflow as tf
import os


# class CCGPT2(tf.keras.Model):
#     def __init__(self, n_vocab, n_embd, n_batch=64, n_layers=1, n_heads=12, learning_rate=1e-3):
#         super(CCGPT2, self).__init__()

#         self.n_vocab = n_vocab
#         self.n_embd = n_embd
#         self.n_batch = n_batch
#         self.n_layers = n_layers
#         self.n_heads = n_heads
#         self.learning_rate = learning_rate

#         self.embedding = tf.keras.layers.Embedding(
#             n_vocab,
#             n_embd,
#             batch_input_shape=(n_batch, None)
#         )
#         self.gru = tf.keras.layers.GRU(
#             1024,
#             return_sequences=True,
#             stateful=True,
#             recurrent_initializer='glorot_uniform'
#         )
#         self.output = tf.keras.layers.Dense(n_vocab)

#     def call(self, x):
#         x = self.embedding(x)
#         x = self.gru(x)
#         return self.output(x)

#     def train_step(self, x, y, optimizer, loss_obj):
#         with tf.GradientTape() as tape:
#             y_hat = self(x, training=True)
#             loss = loss_obj(y, y_hat)

#         gradients = tape.gradient(loss, self.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, self.trainable_variables))

#     def get_optimizer(self):
#         return tf.keras.optimizers.Adam(self.learning_rate)

#     def get_loss(self):
#         return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def build_model(*, n_vocab, n_embd, n_batch):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            n_vocab, n_embd, batch_input_shape=(n_batch, None)),
        tf.keras.layers.GRU(1024, return_sequences=True,
                            stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(n_vocab),
    ])

    return model
