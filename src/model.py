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
        self.decoder_layers = [DecoderLayer(
            d_model=n_embd, n_head=n_head, dff=dff, rate=rate
        ) for _ in range(n_layer)]

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
            x = self.decoder_layers[i](x, training, look_ahead_mask)

        return x

    def get_optimizer(self):
        learning_rate = LRCustomSchedule(self.n_embd)
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        return optimizer

    def get_loss(self):
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none',
        )

        def loss_function(real, pred):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_object(real, pred)
            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

        return loss_function
