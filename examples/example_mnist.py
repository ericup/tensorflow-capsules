import tfcaps as tc
import tensorflow as tf
import numpy as np
from typing import Tuple


# =============================================================================
# Model definition
# =============================================================================


def encoder(inputs: tf.Tensor, classes: int = 10) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Define encoder part
    :param inputs: Inputs for the encoder
    :param classes: Number of classes
    :return:
    """
    i, o = tc.layers.new_io(inputs)
    i(tf.layers.conv2d(o(), filters=256, kernel_size=9, strides=1, activation=tf.nn.relu))
    i(tc.layers.PrimaryConvCaps2D(kernel_size=9, types=32, dimensions=8, strides=2, data_format='channels_last'))
    i(tc.layers.ConvCaps2D(kernel_size=o().shape.as_list()[1:3], types=classes, dimensions=16, name="conv-caps",
                           data_format='channels_last'))
    i(tf.squeeze(o(), axis=(1, 2)))  # shape: [batch, classes, dimensions]
    probabilities = tc.layers.length(o())  # shape: [batch, classes]
    return o(), probabilities


def decoder(inputs: tf.Tensor, shape: np.ndarray) -> tf.Tensor:
    """
    Define decoder part
    :param inputs: Inputs for the decoder.
    :param shape: Shape of a single data point. For MNIST it would be [28, 28, 1].
    :return:
    """
    i, o = tc.layers.new_io(inputs)
    i(tf.layers.dense(o(), 512, activation=tf.nn.relu))
    i(tf.layers.dense(o(), 1024, activation=tf.nn.relu))
    i(tf.layers.dense(o(), np.multiply.reduce(shape), activation=tf.nn.tanh))
    i(tf.reshape(o(), [-1] + shape))
    return o()


def build(inputs: tf.Tensor, labels: tf.Tensor, classes, optimizer: tf.keras.optimizers.Optimizer = None,
          global_step=None, dtype: tf.DType = tf.float32):
    """
    Build model
    :param inputs: Input Tensor. For MNIST with shape [batch, 28, 28, 1]
    :param labels: Label Tensor with shape [batch]
    :param classes: Number of classes
    :param optimizer: An optimizer of choice. Default is Adam.
    :param global_step: Optionally a global step Variable
    :param dtype: Desired data type
    :return:
    """

    training = tf.placeholder_with_default(False, shape=(), name='training')
    if optimizer is None:
        optimizer = tf.train.AdamOptimizer()

    # Encoder
    encoder_out, probabilities = encoder(inputs, classes)
    predictions = tf.argmax(probabilities, axis=-1, output_type=labels.dtype, name="predictions")
    eq = tf.cast(tf.equal(predictions, labels), dtype=dtype)
    correct_predictions = tf.reduce_sum(eq, name="correct_predictions")
    false_predictions = tf.reduce_sum(1 - eq, name="false_predictions")
    accuracy = tf.reduce_mean(eq, name="accuracy")

    # Decoder
    encoder_out_masked_flat = tc.layers.label_mask(encoder_out, labels, predictions, training)
    decoder_out = decoder(encoder_out_masked_flat, inputs.shape.as_list()[1:])  # change this line for dynamic shapes

    # Loss
    loss = tc.losses.capsule_net_loss(inputs, encoder_out, decoder_out, labels, m_minus=.1, m_plus=.9, alpha=.0005)
    train_op = optimizer.minimize(loss, global_step)
    return train_op, training, correct_predictions, false_predictions, accuracy, loss


# =============================================================================
# Generic training setup
# =============================================================================


def prep(data):
    data = data[..., None]
    data /= 127.5
    data -= 1.
    return data


# Load data
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
train_x, test_x = prep(train_x), prep(test_x)

# Create placeholder
x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="x")
y = tf.placeholder(tf.int32, shape=(None,), name="y")

# Build model
train_op, training, correct_predictions, false_predictions, accuracy, loss = build(x, y, 10)

# Create session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train single epoch
batch_size = 128
le = len(train_x) // batch_size
for step in range(le):
    j = step * batch_size
    feed_dict = {
        training: True,
        x: train_x[j:j + batch_size],
        y: train_y[j:j + batch_size],
    }
    _, l, a, cp, fp = sess.run([train_op, loss, accuracy, correct_predictions, false_predictions], feed_dict)
    if (step + 1) % 50 == 0 or step == (le - 1):
        print((step + 1), "of", le, "Loss:", l, "Accuracy:", a, "Correct:", cp, "False:", fp, " " * 42, end="\r")

# Test model
feed_dict = {
    training: False,
    x: test_x,
    y: test_y,
}
a, cp, fp = sess.run([accuracy, correct_predictions, false_predictions], feed_dict)
print("Accuracy:", a, "Correct:", cp, "False:", fp, " " * 42, end="\r")
