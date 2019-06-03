import os
from ..util import force_tuple
from . import functional_ops as ops
import tensorflow as tf
from abc import abstractmethod
from typing import Union


WINDOWS = os.name == 'nt'


# =============================================================================
# Abstract Layer class
# =============================================================================


class Layer:
    def __init__(self, inputs):
        self.inputs = inputs

    def __call__(self, inputs=..., **kwargs) -> Union[object, tf.Tensor]:
        """
        Generic call function for a Layer of any kind.
        Note: 'kwargs' are equivalent to the ones defined in self.__init__
        Behavior:
            - Layer is always applied if `inputs` is defined.
            - If a Layer is to be applied and there are additional keyword arguments a new Layer instance is created and
              it is applied instead. Note that this instance is lost.
            - If a Layer is to be applied and there are no additional keyword arguments it is applied itself.
            - If a Layer is not to be applied a new instance is created and returned
            - New instances always inherit all settings from the current instance.
        :param inputs: (optional) Inputs for this layer. Default setting causes this method to return a new instance.
                       Other settings will apply this layer or a new instance of it.
        :param kwargs: (optional) Keyword arguments, equivalent to the ones from self.__init__
        :return: New instance if `inputs` is not defined, otherwise the resulting Tensor of this layer
        """

        if len(kwargs) != 0:
            # Retrieve class parameters
            params = self.__init__.__code__.co_varnames[1:]
            kwargs_ = {}
            for p in params:
                kwargs_[p] = getattr(self, p)

            # Update with current parameters
            kwargs_.update(kwargs)

            # Create new instance
            instance = self.__init__(kwargs_)
        else:
            instance = self

        if inputs is not ...:
            instance.inputs = inputs
            return instance.call()
        else:
            return instance

    @abstractmethod
    def call(self) -> tf.Tensor:
        pass


# =============================================================================
# Layer classes
# =============================================================================


class BasicPrimaryCapsule(Layer):
    """
    This class represents a Primary Capsule Layer without the convolution operation.

    It basically reshapes and squashes its input accordingly to pave the way for further Capsule Layers.
    IMPORTANT NOTE: For simplicity this Layer does not contain the conv operation. This can be done outside of this
    layer equivalently. Only supports data_format='channel_last'

    Note: Not every setting of 'dimensions' is possible!
    Example: If there is a ConvLayer beforehand that outputs 256 6x6 activation maps,
             then depth % 'dimensions' == 0 must hold.
             The number of capsules then is: 6 * 6 * (256 // 'dimensions')
             for example: 6 * 6 * (256 // 8) = 6 * 6 * 32 = 1152 unique capsules
    """

    def __init__(
            self,
            dimensions,
            inputs=None
    ):
        self.dimensions = dimensions
        super().__init__(inputs)

    def call(self) -> tf.Tensor:
        # (n, height, width, depth)
        height, width, depth = self.inputs.shape[1:4]
        assert depth % self.dimensions == 0
        n = tf.shape(self.inputs)[0]
        return ops.squash(tf.reshape(self.inputs, [n, width * height * (depth // self.dimensions), self.dimensions]))


class PrimaryCapsule(Layer):
    """
    This class represents a Primary Capsule Layer.
    """

    def __init__(
            self,
            types: int,
            dimensions: int,
            kernel_size: Union[int, tuple, list],
            strides: Union[int, tuple, list] = 1,
            padding: str = 'valid',
            data_format='channels_last',
            name: str = None,
            activation: callable = None,
            inputs: tf.Tensor = None
    ):
        """

        :param types: Number of capsule types
        :param dimensions: Number of elements per capsule
        :param kernel_size: Kernel size for 2d convolution
        :param strides: Strides for 2d convolution
        :param padding: Padding for 2d convolution
        :param data_format: Data format
        :param name: Name of this operation
        :param activation: Activation function for 2d convolution
        :param inputs: Inputs for this layer
        """
        self.kernel_size = kernel_size
        self.types = types
        self.dimensions = dimensions
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.name = name
        self.activation = activation
        super().__init__(inputs)

    def call(self) -> tf.Tensor:
        conv = tf.layers.conv2d(self.inputs,
                                filters=self.types * self.dimensions,
                                kernel_size=self.kernel_size,
                                strides=self.strides,
                                padding=self.padding,
                                data_format=self.data_format,
                                activation=self.activation,
                                name=self.name + '_conv' if self.name is not None else None)

        reshape = conv.shape.as_list()
        reshape[0] = tf.shape(conv)[0]
        if self.data_format == 'channels_first':
            reshape[1] = self.dimensions
            reshape.insert(1, self.types)
            axis = 2
        elif self.data_format == 'channels_last':
            reshape[-1] = self.types
            reshape += [self.dimensions]
            axis = -1
        else:
            raise ValueError("Unknown data format: " + str(self.data_format))
        prep = tf.reshape(conv, reshape)  # split depth channel to (types, dimensions)
        flat = ops.flatten(prep, axis=axis)  # flatten, but keep batch size and dimensions
        return ops.squash(flat)  # apply squashing function


class DenseCapsule(Layer):
    def __init__(
            self,
            capsules: int,
            dimensions: int,
            routing: int = 3,
            initializer: Union[tf.keras.initializers.Initializer, tf.Tensor] = None,
            name: str = None,
            parallel_iterations=10,
            swap_memory=False,
            inputs: tf.Tensor = None
    ):
        """
        :param capsules: Number of capsules in this layer
        :param dimensions: Number of dimensions of a single capsule
        :param routing: Number of routing iterations
        :param initializer: Initializer for the variable if one is created. Can either be an initializer object or
                            a Tensor. If it's a Tensor, its shape must be known.
        :param name: Name of this layer
        :param inputs: Inputs of this layer
        """
        super().__init__(inputs)
        self.capsules = capsules
        self.dimensions = dimensions
        self.routing = routing
        self.initializer = initializer
        self.name = name
        self.parallel_iterations = parallel_iterations
        self.swap_memory = swap_memory
        self.num_inputs, self.vec_len_inputs = None, None  # retrieved from self.inputs later on

    def single_capsule_only(self) -> tf.Tensor:
        """
        Special case: Layer outputs a single capsule. Routing is effectively useless in this case, so it is omitted.
        Input Tensor 'self.inputs' of shape (batch, capsules', dimensions')
        :return: Output Tensor of shape (batch, 1, dimensions)
        """
        assert self.capsules == 1
        batch_size = tf.shape(self.inputs)[0]
        capsules_in, dimensions_in = self.inputs.shape.as_list()[1:3]
        inputs = tf.reshape(self.inputs, (batch_size, capsules_in * dimensions_in))
        outputs = tf.layers.dense(inputs, units=self.dimensions, activation=ops.squash,
                                  kernel_initializer=self.initializer, bias_initializer=None,
                                  name=None if self.name is None else self.name + "_dense")
        return tf.reshape(outputs, (batch_size, 1, self.dimensions))

    def call(self) -> tf.Tensor:
        # Special case: A single output capsule
        if self.capsules == 1:
            return self.single_capsule_only()
        # Ordinary case: More than one output capsule
        else:
            self.num_inputs, self.vec_len_inputs = self.inputs.shape.as_list()[1:3]
            with tf.variable_scope("caps" if self.name is None else self.name):
                return ops.dynamic_routing(
                    ops._u_hat_dense(self.inputs, capsules_out=self.capsules, dimensions_out=self.dimensions,
                                     initializer=self.initializer),
                    rounds=self.routing, name=self.name)


class PrimaryConvCaps2D(Layer):
    """
    This layer performs a 2D convolution on a Tensor with shape [batch, height, width, depth] and splits the result
    into capsules and applies the activation function to each capsule.
    The result has the shape [batch, height', width', types, dimensions].
    """

    def __init__(
            self,
            kernel_size: Union[int, tuple, list],
            types: int,
            dimensions: int,
            strides: Union[int, tuple, list] = 1,
            padding: str = 'valid',
            data_format='channels_first',
            name: str = None,
            activation: callable = None,
            inputs: tf.Tensor = None
    ):
        super().__init__(inputs)
        self.kernel_size = force_tuple(kernel_size, 2)
        self.types = types
        self.dimensions = dimensions
        self.strides = strides
        self.padding = padding
        self.name = name
        self.data_format = data_format
        self.activation = activation

    def call(self) -> tf.Tensor:
        assert len(self.inputs.shape.as_list()) == 4

        conv = tf.layers.conv2d(self.inputs,
                                filters=self.types * self.dimensions,
                                kernel_size=self.kernel_size,
                                strides=self.strides,
                                padding=self.padding,
                                data_format=self.data_format,
                                activation=self.activation,
                                name=self.name + '_conv' if self.name is not None else None)

        # Handle different data formats
        if self.data_format == "channels_last":
            # From shape (N, height, width, depth) to (N, height, width, types, depth')
            tmp = tf.concat(
                [tf.expand_dims(elem, axis=3) for elem in tf.split(conv, self.types, axis=3)],
                axis=3
            )
            dims_axis = 3
        elif self.data_format == "channels_first":
            # From shape (N, depth, height, width) to (N, types, depth', height, width)
            tmp = tf.concat(
                [tf.expand_dims(elem, axis=1) for elem in tf.split(conv, self.types, axis=1)],
                axis=1
            )
            dims_axis = 1
        else:
            raise ValueError("Unknown data format:", self.data_format)

        return ops.squash(tmp, axis=dims_axis)


class ConvCaps2D(Layer):
    """
    This Layer natively operates on Tensors of shape: (N, types_in, dimensions_in, height, width).
    With "data_format='channels_last'" Tensors of shape (N, height, width, types_in, dimensions_in)
    can be processed as well, but with computational overhead due to transposing operations.
    """

    def __init__(
            self,
            kernel_size: Union[int, tuple, list],
            types: int,
            dimensions: int,
            name: str,
            data_format: str = 'channels_first',
            routing: int = 3,
            use_bias: bool = True,
            use_conv_bias: bool = True,
            strides: Union[int, tuple, list] = 1,
            padding: str = 'valid',
            dtype: tf.DType = tf.float32,
            inputs: tf.Tensor = None,
    ):
        """

        :param kernel_size:
        :param types:
        :param dimensions:
        :param name:
        :param data_format: Current data format of 'inputs' (also the format of the returned result)
        :param routing:
        :param use_bias:
        :param use_conv_bias:
        :param strides:
        :param padding:
        :param dtype:
        :param inputs:
        """
        super().__init__(inputs)
        self.types = types
        self.dimensions = dimensions
        self.routing = routing
        self.kernel_size = force_tuple(kernel_size, 2)
        self.strides = force_tuple(strides, 2)
        self.padding = padding
        self.name = name
        self.dtype = dtype
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.data_format = data_format

    def call(self) -> tf.Tensor:

        # Handle different data formats
        if self.data_format != "channels_first":
            self.inputs = ops.convert_data_format(self.inputs, data_format="channels_first", channel_types=2)

        params = {
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'bias': self.use_conv_bias,
            'types_out': self.types,
            'dimensions_out': self.dimensions,
            'name': self.name
        }
        u_hat = ops._u_hat_conv2d_light(self.inputs, **params)

        # results.shape: (?, capsules_out, dimensions_out, height, width)
        result = ops.dynamic_routing(u_hat, rounds=self.routing, bias=self.use_bias, name=self.name)

        # Handle different data formats
        if self.data_format != "channels_first":
            result = ops.convert_data_format(result, data_format=self.data_format, channel_types=2)

        return result
