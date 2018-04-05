# -*- coding: utf-8 -*-

from keras import layers
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras import backend as K

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """
    Utility function to apply conv + BN.
    Arguments:
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    Returns:
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def build_inception_graph(img_input, stage5=False):

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = InceptionV3.conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = InceptionV3.conv2d_bn(x, 32, 3, 3, padding='valid')
    x = InceptionV3.conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = InceptionV3.conv2d_bn(x, 80, 1, 1, padding='valid')

    C2 = x = InceptionV3.conv2d_bn(x, 192, 3, 3, padding='valid')

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = InceptionV3.conv2d_bn(x, 64, 1, 1)

    branch5x5 = InceptionV3.conv2d_bn(x, 48, 1, 1)
    branch5x5 = InceptionV3.conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = InceptionV3.conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = InceptionV3.conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = InceptionV3.conv2d_bn(x, 64, 1, 1)

    branch5x5 = InceptionV3.conv2d_bn(x, 48, 1, 1)
    branch5x5 = InceptionV3.conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = InceptionV3.conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = InceptionV3.conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = InceptionV3.conv2d_bn(x, 64, 1, 1)

    branch5x5 = InceptionV3.conv2d_bn(x, 48, 1, 1)
    branch5x5 = InceptionV3.conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = InceptionV3.conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = InceptionV3.conv2d_bn(branch_pool, 64, 1, 1)
    C3 = x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = InceptionV3.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = InceptionV3.conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = InceptionV3.conv2d_bn(x, 192, 1, 1)

    branch7x7 = InceptionV3.conv2d_bn(x, 128, 1, 1)
    branch7x7 = InceptionV3.conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = InceptionV3.conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = InceptionV3.conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = InceptionV3.conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = InceptionV3.conv2d_bn(x, 192, 1, 1)

        branch7x7 = InceptionV3.conv2d_bn(x, 160, 1, 1)
        branch7x7 = InceptionV3.conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = InceptionV3.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = InceptionV3.conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = InceptionV3.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = InceptionV3.conv2d_bn(x, 192, 1, 1)

    branch7x7 = InceptionV3.conv2d_bn(x, 192, 1, 1)
    branch7x7 = InceptionV3.conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = InceptionV3.conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = InceptionV3.conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = InceptionV3.conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = InceptionV3.conv2d_bn(branch_pool, 192, 1, 1)
    C4 = x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = InceptionV3.conv2d_bn(x, 192, 1, 1)
    branch3x3 = InceptionV3.conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = InceptionV3.conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = InceptionV3.conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = InceptionV3.conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = InceptionV3.conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9, 10: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = InceptionV3.conv2d_bn(x, 320, 1, 1)

        branch3x3 = InceptionV3.conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = InceptionV3.conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = InceptionV3.conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = InceptionV3.conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = InceptionV3.conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = InceptionV3.conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = InceptionV3.conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = InceptionV3.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    C5 = x if stage5 else None

    return [None, C2, C3, C4, C5]
