from encoder import get_encoder
from hyperparameters import LRCustomSchedule
from layers.attention import create_masks
from model import CC
from preprocessing import collate_training_dataset

import tensorflow as tf
import json
import os

# if run on an implementation of tensorflow-gpu
# training fails without the stuff below, idk why
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(
        physical_devices[0],
        enable=True,
    )

# retrieve encoder and training dataset
encoder = get_encoder()
dataset = collate_training_dataset(encoder)

# retrieve model hyperparameters
with open(os.path.join('models', 'hparams.json')) as f:
    hparams = json.load(f)

# create model based on hyperparameters
model = CC(*hparams)
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

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy'
)

# configure checkpoint
ckpt_path = './checkpoints/train'
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=5)

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
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    _, combined_mask, _ = create_masks(
        inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = model(inp, tar_inp, True, combined_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)
