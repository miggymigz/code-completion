from .hparams import LRCustomSchedule
from .layers.attention import create_masks
from .layers.decoder import DecoderLayer
from .layers.embedding import positional_encoding

import tensorflow as tf
import time

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='Inputs'),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64, name='Targets'),
]


class CC(tf.keras.Model):
    def __init__(self, *, n_vocab, n_ctx, n_embd, n_head, n_layer, dff=2048, rate=0.1):
        super(CC, self).__init__()

        # model hyperparameters
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        # model layers
        self.embedding = tf.keras.layers.Embedding(n_vocab, n_embd)
        self.pos_encoding = positional_encoding(n_ctx, n_embd)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.dec_layers = [DecoderLayer(
            d_model=n_embd, n_head=n_head, dff=dff, rate=rate
        ) for _ in range(n_layer)]
        self.projection = tf.keras.layers.Dense(n_vocab, dtype=tf.float32)

        # model training stuff
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction='none'
        )
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy'
        )

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

    def create_optimizer(self):
        learning_rate = LRCustomSchedule(self.n_embd)
        with tf.name_scope('optimizer'):
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate,
                beta_1=0.9,
                beta_2=0.98,
                epsilon=1e-9
            )

    def get_loss(self, real, pred):
        with tf.name_scope('loss_layer'):
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = self.loss_object(real, pred)

            with tf.name_scope('loss_masking'):
                mask = tf.cast(mask, dtype=loss_.dtype)
                loss_ *= mask

            loss_ = tf.reduce_sum(loss_, axis=1)
            sequence_avg_loss = loss_ / tf.reduce_sum(mask, axis=1)
            return sequence_avg_loss

    def create_checkpoint_manager(self, checkpoint_path, max_to_keep=5, load_model=True):
        with tf.name_scope('checkpoint_manager'):
            ckpt = tf.train.Checkpoint(model=self, optimizer=self.optimizer)
            self.ckpt_manager = tf.train.CheckpointManager(
                ckpt,
                checkpoint_path,
                max_to_keep=max_to_keep,
            )

            if load_model and self.ckpt_manager.latest_checkpoint:
                ckpt.restore(self.ckpt_manager.latest_checkpoint)
                print('INFO - Latest checkpoint restored.')
            else:
                print('INFO - Will initialize model from scratch.')

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inputs, targets, grad_clip=True, clip_value=2.5):
        mask = create_masks(targets)

        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True, look_ahead_mask=mask)
            loss = tf.reduce_mean(self.get_loss(targets, predictions))

        with tf.name_scope('gradients'):
            gradients = tape.gradient(loss, self.trainable_variables)

            if grad_clip:
                gradients = [tf.clip_by_value(grad, -clip_value, clip_value)
                             for grad in gradients]

            self.optimizer.apply_gradients(
                zip(gradients, self.trainable_variables)
            )

        self.train_loss(loss)
        self.train_accuracy(targets, predictions)

    def fit(self, dataset, epochs=1000):
        for epoch in range(epochs):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for (batch, (inputs, targets)) in enumerate(dataset):
                self.train_step(inputs, targets)

                if batch % 100 == 0:
                    loss = self.train_loss.result()
                    accuracy = self.train_accuracy.result()
                    print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                        epoch + 1, batch, loss, accuracy))

                break

            if (epoch + 1) % 10 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(
                    epoch + 1, ckpt_save_path))

            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, self.train_loss.result(), self.train_accuracy.result()))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
