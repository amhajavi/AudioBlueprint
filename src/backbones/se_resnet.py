import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Activation, Conv1D, Conv2D, Input, Lambda
from tensorflow.keras.layers import BatchNormalization, Flatten, Dense, Reshape, Multiply
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

from tools.ghostvlad import VladPooling

weight_decay = 1e-4



def identity_block_2D(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    block_name = str(stage) + "_" + str(block)
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis,  name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis,  name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis,  name=bn_name_3)(x)


    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])

    x = layers.add([x, input_tensor], name='block_' + block_name)
    x = Activation('relu')(x)
    return x


def conv_block_2D(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    block_name = str(stage) + "_" + str(block)
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    bn_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce/bn'
    x = Conv2D(filters1, (1, 1),
               strides=strides,
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_1)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_1)(x)
    x = Activation('relu')(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    bn_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3/bn'
    x = Conv2D(filters2, kernel_size, padding='same',
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_2)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_2)(x)
    x = Activation('relu')(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    bn_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase/bn'
    x = Conv2D(filters3, (1, 1),
               kernel_initializer='orthogonal',
               use_bias=False,
               kernel_regularizer=l2(weight_decay),
               name=conv_name_3)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_3)(x)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    bn_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj/bn'
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      kernel_initializer='orthogonal',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay),
                      name=conv_name_4)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis,  name=bn_name_4)(shortcut)

    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name = 'fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name = 'fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])

    x = layers.add([x, shortcut], name='block_' + block_name)
    x = Activation('relu')(x)
    return x


def thin_seresnet(input_dim):
    bn_axis = 3
    inputs = Input(shape=input_dim, name='input')

    # ===============================================
    #            Convolution Block 1
    # ===============================================

    x1 = Conv2D(64, (7, 7),
                kernel_initializer='orthogonal',
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                name='conv1_1/3x3_s1')(inputs)

    x1 = BatchNormalization(axis=bn_axis, name='conv1_1/3x3_s1/bn')(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2), strides=(1, 2))(x1)

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(x1, 3, [48, 48, 96], stage=2, block='a', strides=(1, 2))
    x2 = identity_block_2D(x2, 3, [48, 48, 96], stage=2, block='b')

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(x2, 3, [96, 96, 128], stage=3, block='a')
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='b')
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='c')
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(x3, 3, [128, 128, 256], stage=4, block='a')
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='b')
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='c')
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(x4, 3, [256, 256, 512], stage=5, block='a')
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='b')
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='c')
    x = MaxPooling2D((3, 1), strides=(2, 1), name='mpool2')(x5)

    x_fc = Conv2D(512, (7, 1),
                               strides=(1, 1),
                               activation='relu',
                               kernel_initializer='orthogonal',
                               use_bias=True,
                               kernel_regularizer=l2(weight_decay),
                               bias_regularizer=l2(weight_decay),
                               name='x_fc')(x)

    x_k_center = Conv2D(8+2, (7, 1),
                         strides=(1, 1),
                         kernel_initializer='orthogonal',
                         use_bias=True,
                         kernel_regularizer=l2(weight_decay),
                         bias_regularizer=l2(weight_decay),
                         name='gvlad_center_assignment')(x)

    x = VladPooling(k_centers=8, g_centers=2, name='gvlad_pool')([x_fc, x_k_center])

    x = Dense(512, activation='relu',
                           kernel_initializer='orthogonal',
                           use_bias=True,
                           kernel_regularizer=l2(weight_decay),
                           bias_regularizer=l2(weight_decay),
                           name='fc6')(x)


    return inputs, x
