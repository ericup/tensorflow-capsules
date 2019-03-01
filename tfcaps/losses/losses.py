import tensorflow as tf
from ..layers import length as norm
from typing import Union


def margin(
        probabilities: tf.Tensor,
        target: Union[tf.Tensor, int],
        m: float
):
    """
    Calculates the margin for given `probabilities`.
    :param probabilities: The probabilities that are to be evaluated
    :param target: Probability target (either 1 or 0)
    :param m: Margin parameter (usually .9 for target 1 or .1 for target 0)
    :return:
    """
    return target * tf.square(tf.maximum(0., tf.subtract(m, probabilities))) + (1 - target) * tf.square(
        tf.maximum(0., tf.subtract(probabilities, m)))


def margin_loss(
        class_capsules: tf.Tensor,
        labels: tf.Tensor,
        m_minus: float = .1,
        m_plus: float = .9,
        beta: float = 1.,
        dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """
    Compute margin loss with the results of a capsule network 'encoder_out'
    :param class_capsules: Tensor of shape (N, num_classes, dimensions)
    :param labels: True labels
    :param m_minus: Probability margin for capsules that should not be active
    :param m_plus: Probability margin for capsules that should be active
    :param beta: Factor for the loss of the capsules that should not be active
    :param dtype: Desired data type
    :return:
    """
    # Retrieve number of classes
    num_classes = class_capsules.shape[1]

    # Convert labels to one hot encoding from self.labels
    labels_one_hot = tf.cast(tf.one_hot(tf.cast(labels, dtype=tf.int64), num_classes), dtype)

    # Compute 2-norm (with epsilon) of caps output
    caps_norm = norm(class_capsules, axis=-1, keepdims=True)

    # Compute max(0, m^{+} - ||v_k||)^2
    pos_err = tf.reshape(tf.square(tf.maximum(0., tf.subtract(m_plus, caps_norm)), name="present_err"),
                         shape=(-1, num_classes))

    # Compute max(0, ||v_k|| - m^{-})^2
    neg_err = tf.reshape(tf.square(tf.maximum(0., tf.subtract(caps_norm, m_minus)), name="absent_err"),
                         shape=(-1, num_classes))

    # Compute L_k
    pos_err = labels_one_hot * pos_err
    neg_err = beta * (1. - labels_one_hot) * neg_err
    loss_k = tf.add(pos_err, neg_err)

    # Compute margin loss
    return tf.reduce_mean(tf.reduce_sum(loss_k, axis=1), name="margin_loss")


def logarithmic_margin_loss(
        class_capsules,
        labels,
        m_minus=.1,
        m_plus=.9,
        beta=1.,
        dtype=tf.float32
) -> tf.Tensor:
    """
    Compute logarithmic margin loss with the results of a capsule network 'encoder_out'
    :param class_capsules: Tensor of shape (N, num_classes, dimensions)
    :param labels: True labels
    :param m_minus: Probability margin for capsules that should not be active
    :param m_plus: Probability margin for capsules that should be active
    :param beta: Factor for the loss of the capsules that should not be active
    :param dtype: Desired data type
    :return:
    """
    # Retrieve shapes
    batch_size = tf.shape(class_capsules)[0]
    num_classes = class_capsules.shape[1]
    err_shape = (batch_size, num_classes)

    # Convert labels to one hot encoding from self.labels
    labels_one_hot = tf.cast(tf.one_hot(tf.cast(labels, dtype=tf.int64), num_classes), dtype)

    # Compute 2-norm (with epsilon) of caps output
    caps_norm = norm(class_capsules, axis=-1, keepdims=True)

    # Compute
    pos_err = tf.reshape(tf.nn.relu(tf.log(m_plus / (tf.add(caps_norm, 1e-10)))), shape=err_shape)
    neg_err = tf.reshape(tf.nn.relu(tf.log((1 - m_minus) / (tf.subtract(1, caps_norm) + 1e-10))), shape=err_shape)

    # Compute L_k
    pos_err = labels_one_hot * pos_err
    neg_err = beta * (1. - labels_one_hot) * neg_err
    loss_k = tf.add(pos_err, neg_err)

    # Compute margin loss
    return tf.reduce_mean(tf.reduce_sum(loss_k, axis=1), name="margin_loss")


def reconstruction_loss(
        original,
        reconstruction,
        alpha=.0005,
        ssd=True
) -> tf.Tensor:
    """
    Compute scaled (by 'alpha') reconstruction loss with 'original' and 'reconstruction'.
    :param original: Original data points
    :param reconstruction: Reconstruction
    :param alpha: Factor for the final reconstruction loss
    :param ssd: Whether to use the sum of squared distances (SSD)
    :return: Loss Tensor
    """
    a_flat, b_flat = tf.layers.flatten(original), tf.layers.flatten(reconstruction)

    # Compute reconstruction loss
    if ssd:
        # Original: Reduce sum, then mean
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(a_flat - b_flat), axis=1))
    else:
        # Simpler: Reduce mean and choose alpha appropriately
        recon_loss = tf.reduce_mean(tf.square(a_flat - b_flat))

    # Add margin loss with weighted reconstruction loss
    return recon_loss * alpha


def capsule_net_loss(
        encoder_in,
        encoder_out,
        decoder_out,
        labels,
        m_minus=.1,
        m_plus=.9,
        beta=1.,
        alpha=.0005,
        ssd=True,
        dtype=tf.float32
):
    return tf.add(
        margin_loss(
            class_capsules=encoder_out,
            labels=labels,
            m_minus=m_minus,
            m_plus=m_plus,
            beta=beta,
            dtype=dtype
        ),
        reconstruction_loss(
            original=encoder_in,
            reconstruction=decoder_out,
            alpha=alpha,
            ssd=ssd
        )
    )
