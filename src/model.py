from encoder import get_encoder
from preprocessing import collate_training_dataset

import tensorflow as tf
import os

N_VOCAB = 2142
N_EMBD = 768
EPOCHS = 10


def build_model(n_vocab, n_embd):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            n_vocab, n_embd, batch_input_shape=(64, None)),
        tf.keras.layers.GRU(1024, return_sequences=True,
                            stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(n_vocab),
    ])

    return model


def loss(logits, labels):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


encoder = get_encoder()
dataset = collate_training_dataset(encoder)

model = build_model(N_VOCAB, N_EMBD)
model.compile(optimizer='adam', loss=loss)

for x, y in dataset.take(1):
    predictions = model(x)
    print(predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# configure checkpoint
# checkpoint_dir = 'training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     save_weights_only=True,
# )

# do training
# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
