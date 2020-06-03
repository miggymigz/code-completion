from ccompletion.dataset import collate_training_dataset
from ccompletion.hparams import LRCustomSchedule
from ccompletion.layers.attention import create_masks
from ccompletion.model import CC
from ccompletion.preprocessing import get_hparams

import tensorflow as tf
import json
import os
import time


EPOCHS = 1000
BUFFER_SIZE = 10000
BATCH_SIZE = 64
MAX_TO_KEEP = 5


# if run on an implementation of tensorflow-gpu
# training fails without the stuff below, idk why
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(
        physical_devices[0],
        enable=True,
    )


def tf_encode(x, y):
    x.set_shape([None])
    y.set_shape([None])

    return x, y


# retrieve training dataset
dataset = tf.data.Dataset.from_generator(
    collate_training_dataset,
    output_types=(tf.int64, tf.int64),
).map(tf_encode) \
    .cache() \
    .shuffle(BUFFER_SIZE) \
    .padded_batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE)


# retrieve model hyperparameters
hparams = get_hparams()
print(hparams)

# create model based on hyperparameters
model = CC(**hparams)
learning_rate = LRCustomSchedule(hparams['n_embd'])
optimizer = tf.keras.optimizers.Adam(
    learning_rate,
    beta_1=0.9,
    beta_2=0.98,
    epsilon=1e-9
)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction='none'
)


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy'
)

# configure checkpoint
ckpt_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(
    ckpt, ckpt_path,
    max_to_keep=MAX_TO_KEEP
)

# restore latest checkpoint if it exists
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('INFO - Latest checkpoint restored!')


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    combined_mask = create_masks(tar)

    with tf.GradientTape() as tape:
        predictions = model(inp, True, combined_mask)
        loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(tar, predictions)


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(dataset):
        train_step(inp, tar)

        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(
            epoch + 1, ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch + 1, train_loss.result(), train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
