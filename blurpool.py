import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow import keras


class BlurPool1D(Layer):

    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool1D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = np.array([2, 4, 2])
        else:
            raise ValueError

        bk = bk / np.sum(bk)
        bk = np.repeat(bk, input_shape[2])
        bk = np.reshape(bk, (self.kernel_size, 1, input_shape[2], 1))
        blur_init = keras.initializers.constant(bk)

        self.blur_kernel = self.add_weight(name='blur_kernel',
                                           shape=(self.kernel_size, 1, input_shape[2], 1),
                                           initializer=blur_init,
                                           trainable=False)

        super(BlurPool1D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):

        x = K.expand_dims(x, axis=-2)
        x = tf.nn.depthwise_conv2d(x, self.blur_kernel, padding='SAME', strides=(1, self.pool_size, self.pool_size, 1))
        x = K.squeeze(x, axis=-2)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.floor(input_shape[1] / 2)), input_shape[2]

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pool_size': self.pool_size,
            'kernel_size': self.kernel_size
        })
        return config
