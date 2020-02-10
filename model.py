from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from blurpool import AverageBlurPooling1D, MaxBlurPooling1D, BlurPool1D


class ResConvBlock(object):
    def __init__(self,
                 shape: int,
                 dims: int,
                 name: str,
                 pooling=None,
                 kernel_regularizer=None,
                 kernel_initializer: str = 'glorot_uniform'):

        self.name = name
        self.pooling = pooling
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer
        self.dims = dims

        self.bn_1 = BatchNormalization(name='%s_bn_1' % name)
        self.act_1 = Activation('relu', name='%s_act_1' % name)
        self.bn_2 = BatchNormalization(name='%s_bn_2' % name)
        self.act_2 = Activation('relu', name='%s_act_2' % name)

        self.conv_1 = Conv1D(dims, shape,
                             name='%s_conv_1' % name,
                             use_bias=False,
                             padding='same',
                             kernel_regularizer=kernel_regularizer,
                             kernel_initializer=kernel_initializer)

        self.conv_2 = Conv1D(dims, shape,
                             name='%s_conv_2' % name,
                             strides=1 if pooling != 'stride_noaa' else 2,
                             use_bias=True,
                             padding='same',
                             kernel_regularizer=kernel_regularizer,
                             kernel_initializer=kernel_initializer)

        if pooling == 'stride':
            self.pool = BlurPool1D(name='%s_pool' % name)
            self.skip_pool = None
        elif pooling == 'max':
            self.pool = MaxBlurPooling1D(name='%s_pool' % name)
            self.skip_pool = None
        elif pooling == 'avg':
            self.pool = AverageBlurPooling1D(name='%s_pool' % name)
            self.skip_pool = None
        elif pooling == 'stride_noaa':
            self.pool = None
            self.skip_pool = MaxPooling1D(name='%s_skip_pool' % name)
        else:
            self.pool = None
            self.skip_pool = None

        self.residual = Add(name='%s_residual' % self.name)

        self.skip_project = Conv1D(self.dims, 1,
                                   use_bias=True,
                                   name='%s_skip_project' % self.name,
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   kernel_regularizer=self.kernel_regularizer)

    def __call__(self, x):

        # Normal Path
        xn = x
        xn = self.bn_1(xn)
        xn = self.act_1(xn)

        xa = xn

        xn = self.conv_1(xn)
        xn = self.bn_2(xn)
        xn = self.act_2(xn)
        xn = self.conv_2(xn)

        if x.shape[-1] != xn.shape[-1]:
            xs = self.skip_project(xa)
        else:
            xs = x

        if self.pooling == 'stride_noaa':
            xs = self.skip_pool(xs)

        y = self.residual([xn, xs])
        if self.pooling is not None and self.pooling != 'stride_noaa':
           y = self.pool(y)

        return y


def residual_model(dims: int = 32,
                   pooling=None,
                   kernel_regularizer=None,
                   seqlen: int = 2000,
                   n_feat: int = 10,
                   kernel_initializer: str = 'glorot_uniform'):

    layer_input = Input((seqlen, n_feat))
    layer_discount = Input((1,))

    def create_d_binary_crossentropy(d):
        def d_binary_crossentropy(y_true, y_pred):
            x = d * K.binary_crossentropy(y_true, y_pred)
            return K.mean(x, axis=-1)

        return d_binary_crossentropy

    def bce(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred))

    metrics = {'cls': ['acc', bce]}
    loss_fns = [create_d_binary_crossentropy(layer_discount)]

    def unique(start):
        i = start
        while True:
            yield str(i)
            i += 1

    x = layer_input

    conv1 = Conv1D(dims, 9,
                   strides=1 if pooling != 'stride_noaa' else 2,
                   padding='same',
                   name='0_conv_1',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer)
    bn1 = BatchNormalization(name='0_bn')
    act1 = ReLU(name='0_relu')

    if pooling == 'avg':
        pool = AverageBlurPooling1D(name='0_pool')
    elif pooling == 'max':
        pool = MaxBlurPooling1D(name='0_pool')
    elif pooling == 'stride':
        pool = BlurPool1D(name='0_pool')
    else:
        pool = None

    conv2 = Conv1D(dims, 3,
                   padding='same',
                   name='0_conv_2',
                   kernel_regularizer=kernel_regularizer,
                   kernel_initializer=kernel_initializer)

    namer = unique(start=1)

    res_1 = ResConvBlock(shape=3,
                         dims=dims,
                         name=next(namer),
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer,
                         pooling=pooling)

    res_2 = ResConvBlock(shape=3,
                         dims=dims * 2,
                         name=next(namer),
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer,
                         )

    res_3 = ResConvBlock(shape=3,
                         dims=dims * 2,
                         name=next(namer),
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer,
                        )

    res_4 = ResConvBlock(shape=3,
                         dims=dims * 2,
                         name=next(namer),
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer,
                         pooling=pooling)

    res_5 = ResConvBlock(shape=3,
                         dims=dims * 4,
                         name=next(namer),
                         kernel_regularizer=kernel_regularizer,
                         kernel_initializer=kernel_initializer)

    gmp = GlobalMaxPooling1D(name='gmp')
    cls = Dense(1, activation='sigmoid', name='cls', use_bias=False)

    # Forward Pass
    x = conv1(x)
    x = bn1(x)
    x = act1(x)
    if pool is not None:
        x = pool(x)
    x = conv2(x)

    x = res_1(x)
    x = res_2(x)
    x = res_3(x)
    x = res_4(x)
    x = res_5(x)

    x = gmp(x)

    x = cls(x)

    salience = K.abs(K.gradients(x, layer_input)[0])

    model = Model(inputs=[layer_input, layer_discount], outputs=x)
    model.summary()
    return model, loss_fns, metrics, salience
