import tensorflow as tf
import numpy as np
import os
from ..util import force_tuple

WINDOWS = os.name == 'nt'


def new_io(inputs, return_list=False):
    """
    Constructs simple input/output interface that can store and retrieve values from a list.
    This can be handy for compact code in a TensorFlow model definition.
    The input function 'i(v)' tries to call 'v.call()' and adds the returned value to list. If it can't, 'v' is added.
    The input function 'i(v)' tries to call 'v(o())' when 'v' is callable, 'v.inputs' and 'v.call' exist and
    'v.inputs' is 'None' or '...'.
    :param inputs: initial input
    :param return_list: Return value list along the i/o functions
    :return: tuple (input_function, output_function) if 'return_list' is True;
             (value_list, input_function, output_function) otherwise.
    """
    a = [inputs]

    def i(v):
        if hasattr(v, "call"):
            if hasattr(v, "inputs") and (v.inputs is None or v.inputs is ...) and callable(v):
                v = v(o())
            else:
                v = v.call()
        return a.append(v)

    def o(v=-1):
        return a[v]

    if return_list:
        return a, i, o
    return i, o


def label_mask(inputs: tf.Tensor, y_true: tf.Tensor, y_pred: tf.Tensor = None, training: tf.Tensor = None,
               flatten: bool = True):
    """
    Mask `inputs` with the given label information.
    If the predicted label information `y_pred` is provided along with the bool `training` this op uses the
    ground truth `y_true` when `training` yields True and the predictions `y_pred` otherwise.
    :param inputs: Input Tensor. Usually the output of a capsule encoder with shape [batch, classes, depth]
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param training: A tf.placeholder that indicates whether it's training time or not
    :param flatten: Whether to return a flattened Tensor or not
    :return: Result Tensor of shape [batch, classes * depth] if `flatten` is True,
             [batch, classes, depth] otherwise
    """

    # If is training: use latent vector with index = correct class
    # Else: use latent vector with greatest 2-norm
    if y_pred is None:
        reconstruction_targets = tf.cast(y_true, tf.int64)
    else:
        assert training is not None
        reconstruction_targets = tf.cond(training,
                                         lambda: tf.cast(y_true, tf.int64),
                                         lambda: tf.cast(y_pred, tf.int64))

    # Generate a one hot enc. mask of shape (batch, classes)
    num_classes = tf.shape(inputs)[1]
    mask = tf.expand_dims(tf.cast(tf.one_hot(reconstruction_targets,
                                             depth=num_classes,
                                             name="mask"), inputs.dtype), axis=-1)

    masked = tf.multiply(inputs, mask, name="decoder_input")
    if flatten:
        return tf.layers.flatten(masked)
    return masked


def capsule_expand(inputs: tf.Tensor, labels: tf.Tensor, num_capsules: int):
    """
    Constructs a Tensor of shape (N, num_capsules, dimensions) where only one of the 'num_capsules' vectors per batch
    element is nonzero, based on 'inputs' of shape (N, dimensions) and 'labels'.
    :param inputs: Tensor of shape (N, dimensions)
    :param labels: Flat Tensor with labels
    :param num_capsules: Number of desired capsules for the output
    :return: Tensor of shape (N, num_capsules, dimensions)
    """
    # (N, dimensions) -> (N, 1, dimensions) -> (N, num_capsules, dimensions)
    tmp = tf.tile(tf.expand_dims(inputs, 1), multiples=(1, num_capsules, 1))
    return label_mask(tmp, y_true=labels)


def capsule_extract(inputs: tf.Tensor, labels: tf.Tensor):
    """
    Extract single capsule vector for each element of batch 'inputs'.
    Numpy equivalent: return inputs[range(len(inputs)), labels]
    :param inputs: Tensor of shape (N, num_classes, len_vectors)
    :param labels: Flat Tensor with labels
    :return: Tensor of shape (N, len_vectors)
    """
    return tf.gather_nd(inputs, indices=tf.concat(
        (tf.range(0, tf.shape(labels)[0], dtype=tf.int32)[:, None], tf.cast(labels[:, None], dtype=tf.int32)),
        axis=1))


def vary_caps_vector(capsule_vectors: tf.Tensor, start: float, stop: float, delta: float, labels=None):
    """

    :param capsule_vectors: Tensor (N, 1, dimensions)
    :param start: Start value (inclusive)
    :param stop: Stop value (inclusive)
    :param delta: Space between values
    :param labels: (optional) Labels will be modified to be consistent with returned batch
    :return: Tensor (M, 1, dimensions) with M >= N along with modified `labels` if `labels` was not None
    """
    with tf.variable_scope("capsule_tweaks"):
        if len(capsule_vectors.shape) == 2:
            capsule_vectors = tf.expand_dims(capsule_vectors, 1)

        dims = capsule_vectors.shape.as_list()[-1]
        r = np.arange(start, stop + 1e-7, delta, dtype=np.float32)
        rlen = r.shape[0]

        # Example:
        # diags for r=[.25, .5] with dims=3:
        # [.25, 0, 0]
        # [0, .25, 0]
        # [0, 0, .25]
        # [.5, 0, 0]
        # [0, .5, 0]
        # [0, 0, .5]

        diags = tf.reshape(tf.map_fn(lambda x: tf.diag(tf.cast([x] * dims, dtype=tf.float32)), r,
                                     dtype=tf.float32), (1, dims * rlen, dims))
        inputs = tf.tile(capsule_vectors, (1, dims * rlen, 1))  # shape: (N, dims * rlen, dims)
        outputs = tf.reshape(tf.add(inputs, diags), (-1, 1, dims))

        if labels is not None:
            labels = tf.reshape(tf.tile(tf.expand_dims(labels, axis=1), (1, dims * rlen)), (-1,))
            return outputs, labels

        return outputs


def length(inputs, axis=-1, keepdims=False, epsilon=1e-10, name=None) -> tf.Tensor:
    """
    Computes the vector length (2-norm) along specified ´axis´ of given Tensor ´inputs´.
    Optionally an epsilon can be added to the squared norm before the square root is computed.
    """
    with tf.name_scope(name, default_name="norm"):
        if epsilon is None:
            return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=keepdims))
        else:
            return tf.sqrt(tf.add(tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=keepdims), epsilon))


def convert_data_format(inputs: tf.Tensor, data_format, channel_types=1):
    """
    This function toggles data_format.

    Examples for channel_types = 1:
        If 'data_format' is "channels_first":
            (N, height, width, [...,] depth) -> (N, depth, height, width, [...,])
        If 'data_format' is "channels_last":
            (N, depth, height, width, [...,]) -> (N, height, width, [...,] depth)

    Examples for channel_types = 2:
        If 'data_format' is "channels_first":
            (N, height, width, [...,] depth0, depth1) -> (N, depth0, depth1, height, width, [...,])
        If 'data_format' is "channels_last":
            (N, depth0, depth1, height, width, [...,]) -> (N, height, width, [...,] depth0, depth1)

    :param inputs: Tensor with shape (N, height, width, [...,] *depths). Note: Validity of shape depends
                   on 'channel_types'.
    :param data_format: Target data format
    :param channel_types: Number of channel types. Layers like tf.layers.conv2d work with 'channel_types=1'.
    :return: Tensor
    """

    inputs_perm = list(range(len(inputs.shape.as_list())))
    if data_format == "channels_first":
        # (N, height, width, [...,] *depths) -> (N, *depths, height, width, [...,])
        for u, v in zip(list(range(1, channel_types + 1)), list(range(-channel_types, 0))):
            inputs_perm.insert(u, inputs_perm[v])
            del inputs_perm[v]
    elif data_format == "channels_last":
        # (N, *depths, height, width, [...,]) -> (N, height, width, [...,] *depths)
        for _ in range(channel_types):
            inputs_perm += inputs_perm[1:2]
            del inputs_perm[1]
    else:
        raise ValueError("Unknown data format: " + str(data_format))

    # Transpose and return
    return tf.transpose(inputs, perm=inputs_perm)


def flatten(inputs: tf.Tensor, axis=-1):
    """
    This function flattens any Tensor with shape (batch [, ...], dimensions [, ...]) to
    shape (batch, ?, dimensions) with 'axis!=0'.
    If 'axis=0' the output shape is (batch, ?)
    Example:
        A convolutional capsule Tensor with shape (batch, height, width, types, dimensions)
        would be reshaped to (batch, height*width*types, dimensions) with 'axis=-1'
    :param inputs: Tensor
    :param axis: (optional) Defines the axis that is not to be flattened, i.e. the axis of 'dimensions'.
    :return: Tensor
    """

    # For axis=0 perform normal flatten op
    if axis == 0:
        return tf.layers.flatten(inputs)

    batch_size = tf.shape(inputs)[0]

    if axis != -1:
        perm = list(range(len(inputs.shape.as_list())))
        perm_ = list(perm)
        del perm_[axis]
        perm_ += [axis]
        if perm != perm_:
            inputs = tf.transpose(inputs, perm)

    dimensions = inputs.shape.as_list()[-1]
    intermediates = inputs.shape.as_list()[1:-1]
    return tf.reshape(inputs, shape=(batch_size, np.multiply.reduce(intermediates), dimensions))


def entropy(pk, axis=0) -> tf.Tensor:
    """
    Compute the entropy of a distribution for given probability values.
    The values given by 'pk' are expected to sum to 1.
    :param pk: Defines the (discrete) distribution.
               (e.g. [.3, .2, .5] with axis=0 yields 1.02965)
               (e.g. [[.4, .6], [.2, .8]] with axis=1 yields [0.67, 0.5])
    :param axis: Axis on which the entropy is to be computed.
    :return: Result Tensor
    """
    return -tf.reduce_sum(pk * tf.log(pk), axis=axis)


def squash(inputs, axis=-1, epsilon=1e-7, name=None) -> tf.Tensor:
    """
    As described in "Dynamic Routing Between Capsules" (Sabour et al.), Equation 1
    Squash function squashes length (2-norm) of vectors to interval [0, 1]
    :param inputs: input Tensor
    :param axis: the 'dimensions' axis that represents the individual capsules
    :param epsilon: (optional) epsilon value
    :param name: (optional) name of this op
    :return: Tensor
    """
    with tf.name_scope(name=name, default_name='squash'):
        norm_square = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
        return norm_square / (1. + norm_square) * inputs / tf.sqrt(norm_square + epsilon)


def dynamic_routing(u_hat: tf.Tensor, rounds=3, bias: bool = False, name: str = None,
                    dtype=tf.float32) -> tf.Tensor:
    """
    As described in "Dynamic Routing Between Capsules" (Sabour et al.), Procedure 1: Routing algorithm
    This implementation supports 1D and 2D routing.
    :param u_hat: Tensor of shape (N, capsules_in, capsules_out, dimensions_out[, height, width])
    :param rounds: number of iterations for the routing algorithm
    :param bias: Whether to use biases
    :param name: Name of this op
    :param dtype: Desired data type
    :return: Tensor v of shape (?, capsules_out, dimensions_out[, height, width])
    """
    with tf.name_scope("dynamic_routing_between_capsules"):
        # Hardcoded settings
        capsules_out_axis = 2
        dims_axis = 3
        stop_gradient = True

        # Handle shape variables
        # (N, capsules_in, capsules_out, dimensions_out[, height, width]) u_hat
        # (N, capsules_in, capsules_out, 1             [, height, width]) b
        # (1, 1          , capsules_out, dimensions_out[, 1     , 1     ]) biases
        # û shape
        u_hat_shape = u_hat.shape.as_list()[1:]
        # b shape
        b_shape = tf.shape(u_hat)
        b_shape[dims_axis]._value = 1
        # biases shape
        biases_shape = np.ones(len(u_hat_shape) + 1)
        biases_shape[dims_axis] = u_hat_shape[dims_axis - 1]
        biases_shape[capsules_out_axis] = u_hat_shape[capsules_out_axis - 1]

        # Biases
        if bias:
            biases = tf.get_variable('bias' if name is None else name + '_bias', shape=biases_shape)
        else:
            biases = 0.

        # Routing
        if rounds == 0:
            s = tf.reduce_sum(u_hat, axis=1, keepdims=True) + biases
            v = squash(s, axis=dims_axis)
        else:
            b = tf.zeros(b_shape, dtype=dtype)
            u_hat_stop_gradient = tf.stop_gradient(u_hat, name='u_hat_stop_gradient')
            v = None

            for r in range(rounds):
                # Mask u_hat from the gradient generator
                # (N, capsules_in, capsules_out, dimensions_out[, height, width]) u_hat_
                u_hat_ = u_hat_stop_gradient if r < rounds - 1 and stop_gradient else u_hat

                # c_i <-- softmax(b_i)
                # (N, capsules_in, capsules_out, 1[, height, width]) b
                # (N, capsules_in, capsules_out, 1[, height, width]) c
                c = tf.nn.softmax(b, axis=capsules_out_axis)

                # s_j <-- sum_i c_{i,j} û_{j|i}
                # (N, capsules_in, capsules_out, dimensions_out[, height, width]) u_hat_
                # (N, capsules_in, capsules_out, 1             [, height, width]) c
                # (N, 1,           capsules_out, dimensions_out[, height, width]) s
                s = tf.reduce_sum(tf.multiply(c, u_hat_), axis=1, keepdims=True) + biases

                # v_j <-- squash(s_j)
                # (N, 1, capsules_out, dimensions_out[, height, width]) s
                # (N, 1, capsules_out, dimensions_out[, height, width]) v
                v = squash(s, axis=dims_axis)

                # Calculation of b only required if there is another routing iteration
                if r < rounds - 1:
                    # b_{i,j} <-- b_{i,j} + û_{j|i} v_j
                    # (N, capsules_in, capsules_out, dimensions_out[, height, width]) u_hat_
                    # (N, 1,           capsules_out, dimensions_out[, height, width]) v
                    # (N, capsules_in, capsules_out, 1             [, height, width]) b
                    b += tf.reduce_sum(tf.multiply(u_hat_, v), axis=dims_axis, keepdims=True)

        return tf.squeeze(v, axis=1)  # output shape: (?, capsules_out, dimensions_out[, height, width])


def _u_hat_conv2d_light(inputs, kernel_size, strides, padding, bias, types_out, dimensions_out,
                        name, **conv2d_kwargs):
    """

    :param inputs:
    :param kernel_size:
    :param strides:
    :param padding:
    :param bias:
    :param types_out:
    :param dimensions_out:
    :param name:
    :param conv2d_kwargs:
    :return:
    """
    # Shapes
    # Former solution was not compatible with varying image sizes
    # batch_size = tf.shape(inputs)[0]
    # types_in, dimensions_in, height, width = inputs.shape.as_list()[1:]
    ts = tf.shape(inputs)
    batch_size, height, width = ts[0], ts[3], ts[4]
    types_in, dimensions_in = inputs.shape.as_list()[1:3]

    # No dedicated weights for input types
    # Reshape
    # (N, types_in,  dimensions_in, height, width) inputs
    # (N * types_in, dimensions_in, height, width) inputs
    inputs = tf.reshape(inputs, tf.stack((batch_size * types_in, dimensions_in, height, width)))

    # Calculate û
    # (N * types_in, dimensions_in, height, width) inputs
    # (N * types_in, types_out * dimensions_out, height_out, width_out) u_hat_packed
    if WINDOWS:  # remove when windows bug is fixed
        inputs = convert_data_format(inputs, data_format='channels_last', channel_types=1)
    u_hat_packed = tf.layers.conv2d(
        inputs=inputs,
        filters=types_out * dimensions_out,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format='channels_last' if WINDOWS else 'channels_first',
        use_bias=bias,
        name=name if name is None else name + "_conv2d",
        **conv2d_kwargs
    )
    if WINDOWS:
        u_hat_packed = convert_data_format(u_hat_packed, data_format='channels_first', channel_types=1)

    # Unpack
    # (N * types_in, types_out * dimensions_out, height_out, width_out) u_hat_packed
    # (N,  types_in, types_out,  dimensions_out, height_out, width_out) u_hat
    ts = tf.shape(u_hat_packed)
    height_out, width_out = ts[-2], ts[-1]
    u_hat = tf.reshape(u_hat_packed, tf.stack((batch_size, types_in, types_out, dimensions_out, height_out, width_out)))
    return u_hat


def _u_hat_conv2d_light_alt(inputs, kernel_size, strides, padding, bias, types_out,
                            dimensions_out, name, **conv2d_kwargs):
    """

    :param inputs:
    :param kernel_size:
    :param strides:
    :param padding:
    :param bias:
    :param types_out:
    :param dimensions_out:
    :param name:
    :param conv2d_kwargs:
    :return:
    """
    # Shapes
    batch_size = tf.shape(inputs)[0]
    inputs_shape = inputs.shape.as_list()
    types_in, dimensions_in, height, width = inputs_shape[1:]

    # Transposes
    # Note: conv3d expects (batch, channels, depth, height, width)
    # (N, types_in, dimensions_in, height, width) inputs
    # (N, dimensions_in, types_in, height, width) inputs
    inputs_perm = list(range(len(inputs_shape)))
    inputs_perm[1:3] = [2, 1]
    inputs = tf.transpose(inputs, perm=inputs_perm)
    u_hat_perm = list(range(len(inputs_shape) + 1))
    u_hat_perm[1:4] = [3, 1, 2]

    # Convert params
    def convert_tuple(tup):
        """ Make tuple param compatible with conv3d """
        if isinstance(tup, (list, tuple)):
            assert len(tup) < 3
        return (1,) + force_tuple(kernel_size, 2)

    kernel_size = convert_tuple(kernel_size)
    strides = convert_tuple(strides)

    # Calculate u_hat
    # (N, dimensions_in, types_in, height, width) inputs
    # (N, types_out * dimensions_out, types_in, height_out, width_out) u_hat_packed
    if WINDOWS:  # remove when windows bug is fixed
        inputs = convert_data_format(inputs, data_format='channels_last', channel_types=1)
    u_hat_packed = tf.layers.conv3d(
        inputs=inputs,
        filters=types_out * dimensions_out,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format='channels_last' if WINDOWS else 'channels_first',
        use_bias=bias,
        name=name if name is None else name + "_conv3d",
        **conv2d_kwargs
    )
    if WINDOWS:
        u_hat_packed = convert_data_format(u_hat_packed, data_format='channels_first', channel_types=1)

    # Unpack
    # (N, types_out * dimensions_out, types_in, height_out, width_out) u_hat_packed
    # (N, types_out,  dimensions_out, types_in, height_out, width_out) u_hat_mixed
    # (N, types_in,  types_out, dimensions_out, height_out, width_out) u_hat
    height_out, width_out = u_hat_packed.shape.as_list()[-2:]
    u_hat_mixed = tf.reshape(u_hat_packed, (batch_size, types_out, dimensions_out, types_in, height_out, width_out))
    u_hat = tf.transpose(u_hat_mixed, perm=u_hat_perm)
    return u_hat


def _u_hat_dense(inputs, capsules_in, dimensions_in, capsules_out, dimensions_out, weights=None, initializer=None,
                 nested=False, parallel_iterations=10, swap_memory=False, dtype=tf.float32):
    if weights is None:
        if initializer is None:
            initializer = tf.random_normal(
                shape=(capsules_out, capsules_in, dimensions_out, dimensions_in),
                stddev=.1,
                dtype=dtype,
                name="W_rand_normal"
            )
            weights = tf.get_variable(initializer=initializer, name="W", trainable=True)

    if nested:
        return _u_hat_dense_nested(inputs, weights=weights, parallel_iterations=parallel_iterations,
                                   swap_memory=swap_memory)
    else:
        return _u_hat_dense_dist(inputs, weights=weights, capsules_out=capsules_out,
                                 parallel_iterations=parallel_iterations, swap_memory=swap_memory)


def _u_hat_dense_dist(inputs, weights, capsules_out, parallel_iterations=10, swap_memory=False):
    """
    This method returns a tensor that contains all u_hat values.

    Comment on the implementation:
        Note that TensorFlow does not support broadcasting in this particular case at this point in time.
        This method is essentially a compromise between memory overhead due to manual
        broadcasting and parallelization via tf.map_fn.
    :param inputs:
    :param weights:
    :param capsules_out:
    :param parallel_iterations: The number of iterations that run in parallel.
    :param swap_memory: (optional) True enables GPU-CPU memory swapping.
    :return: Tensor with shape: (N, num_capsules_in, num_capsules_out, num_dimensions_out)
    """
    # get input shape from (N, num_capsules_in, num_dimensions_out) to
    # (N, capsules_out, capsules_in, dimensions_out, 1) by tiling
    inputs_tiled = tf.tile(
        input=tf.expand_dims(tf.expand_dims(inputs, axis=-1), axis=1),
        multiples=[1, capsules_out, 1, 1, 1]
    )

    # Apply W to every tensor in current batch
    return tf.squeeze(
        tf.transpose(tf.map_fn(lambda x: tf.matmul(weights, x), inputs_tiled, parallel_iterations=parallel_iterations,
                               swap_memory=swap_memory),
                     perm=(0, 2, 1, 3, 4)),
        axis=-1
    )


def _u_hat_dense_nested(inputs, weights, parallel_iterations=10, swap_memory=False):
    """
    This method returns a tensor that contains all u_hat values.

    Comment on the implementation:
        Note that TensorFlow does not support broadcasting in this particular case at this point in time.
        This method uses nested tf.map_fn to offer high potential for
        parallelization and avoids manual broadcasting entirely.
        Use with caution, not all machines are able to handle this. It may cause bottlenecking.
    :param inputs:
    :param weights:
    :param parallel_iterations: The number of iterations that run in parallel. NOTE: The actual deployed number is
                                defined as: int(sqrt(parallel_iterations))**2
    :param swap_memory: (optional) True enables GPU-CPU memory swapping.
    :return: Tensor with shape: (N, num_capsules_in, num_capsules_out, num_dimensions_out)
    """
    parallel_iterations = int(np.sqrt(parallel_iterations))

    # get input shape from (N, num_capsules_in, num_dimensions_out) to (N, num_capsules_in, num_dimensions_out, 1)
    inputs_ = tf.expand_dims(inputs, axis=-1)

    # Apply every w with shape (vec_len_outputs, vec_len_inputs) to every x with shape (len_vectors, 1)
    return tf.squeeze(
        tf.transpose(tf.map_fn(
            lambda x: tf.map_fn(lambda w: tf.matmul(w, x), weights, parallel_iterations=parallel_iterations,
                                swap_memory=swap_memory), inputs_, parallel_iterations=parallel_iterations,
            swap_memory=swap_memory), perm=(0, 2, 1, 3, 4)),
        axis=-1
    )
