from .attention import MultiHeadAttention
from .feedforward import point_wise_feed_forward_network

import tensorflow as tf


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.attn = MultiHeadAttention(d_model, n_head)
        self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn_dropout = tf.keras.layers.Dropout(rate)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, look_ahead_mask):
        # (batch_size, target_seq_len, d_model)
        attn_out, _ = self.attn(x, x, x, look_ahead_mask)
        attn_out = self.attn_dropout(attn_out, training=training)
        attn_out = self.attn_norm(attn_out + x)

        ffn_out = self.ffn(attn_out)  # (batch_size, target_seq_len, d_model)
        ffn_out = self.ffn_dropout(ffn_out, training=training)
        # (batch_size, target_seq_len, d_model)
        ffn_out = self.ffn_norm(ffn_out + attn_out)

        return ffn_out
