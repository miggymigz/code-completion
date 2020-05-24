from hyperparameters import LRCustomSchedule
from layers.embedding import positional_encoding
from layers.decoder import DecoderLayer

import tensorflow as tf


class CC(tf.keras.Model):
    def __init__(self, *, n_vocab, n_ctx, n_embd, n_head, n_layer, dff=2048, rate=0.1):
        super(CC, self).__init__()

        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        self.embedding = tf.keras.layers.Embedding(n_vocab, n_embd)
        self.pos_encoding = positional_encoding(n_ctx, n_embd)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dec_layers = [DecoderLayer(
            d_model=n_embd, n_head=n_head, dff=dff, rate=rate
        ) for _ in range(n_layer)]
        self.projection = tf.keras.layers.Dense(n_vocab, dtype=tf.float32)

    def call(self, x, training, look_ahead_mask):
        # retrieve sequence length
        seq_len = tf.shape(x)[1]

        # add token embedding and positional encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.n_embd, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        # add dropout
        x = self.dropout(x, training=training)

        # go through all of the layers
        for i in range(self.n_layer):
            x = self.dec_layers[i](x, training, look_ahead_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        x = self.projection(x)

        return x
