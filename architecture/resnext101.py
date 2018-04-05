
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Add
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D

def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True):
    """
    This block is the one with no convolutional layer at its shortcut branch
    :param input_tensor: block input.
    :param kernel_size: kernel size of the convolutional layer
    :param filters: filters of the convolutional layer
    :param stage: layer name
    :param block: block name
    :param use_bias: bias the layer
    :return: block as layer sequence
    """

    filter1, filter2, filter3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True):
    """
    This block is the one with convolutional layer at its shortcut branch
    :param input_tensor: block input.
    :param kernel_size: kernel size of the convolutional layer
    :param filters: filters of the convolutional layer
    :param stage: layer name
    :param block: block name
    :param strides: stride of the convolutional layer
    :param use_bias: bias the layer
    :return: block as layer sequence
    """

    filter1, filter2, filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNorm(axis=3, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(axis=3, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu', name='res' + str(stage) + block + '_out')(x)

    return x

def build_resnext_graph(input_image, stage5=False):

    """
    Model generator
    :param input_image: model input
    :param stage5: enables or disables the last stage
    :return: returns the network stages
    """

    # Stage 1
    x = ZeroPadding2D((3, 3))(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    C1 = x = MaxPooling2D((2, 2), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [128, 128, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [128, 128, 256], stage=2, block='b')
    C2 = x = identity_block(x, 3, [128, 128, 256], stage=2, block='c')

    # Stage 3
    x = conv_block(x, 3, [256, 256, 512], stage=3, block='a')
    x = identity_block(x, 3, [256, 256, 512], stage=3, block='b')
    x = identity_block(x, 3, [256, 256, 512], stage=3, block='c')
    C3 = x = identity_block(x, 3, [256, 256, 512], stage=3, block='d')

    # Stage 4
    x = conv_block(x, 3, [512, 512, 1024], stage=4, block='a')

    for i in range(22):
        x = identity_block(x, 3, [512, 512, 1024], stage=4, block=chr(98 + i))
    C4 = x

    # Stage 5
    if stage5:
        x = conv_block(x, 3, [1024, 1024, 2048], stage=5, block='a')
        x = identity_block(x, 3, [1024, 1024, 2048], stage=5, block='b')
        C5 = identity_block(x, 3, [1024, 1024, 2048], stage=5, block='c')
    else:
        C5 = None

    return [C1, C2, C3, C4, C5]


class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.
    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=False)