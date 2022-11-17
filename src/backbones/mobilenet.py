# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf

import tensorflow.keras.layers as layers


def _depthwise_conv_block(
    inputs,
    pointwise_conv_filters,
    alpha,
    depth_multiplier=1,
    strides=(1, 1),
    block_id=1,
):
    """Adds a depthwise convolution block.
    A depthwise convolution block consists of a depthwise conv,
    batch normalization, relu6, pointwise convolution,
    batch normalization and relu6 activation.
    Args:
      inputs: Input tensor of shape `(rows, cols, channels)` (with
        `channels_last` data format) or (channels, rows, cols) (with
        `channels_first` data format).
      pointwise_conv_filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the pointwise convolution).
      alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
      depth_multiplier: The number of depthwise convolution output channels
        for each input channel. The total number of depthwise convolution
        output channels will be equal to `filters_in * depth_multiplier`.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any `dilation_rate`
        value != 1.
      block_id: Integer, a unique identification designating the block number.
        # Input shape
      4D tensor with shape: `(batch, channels, rows, cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(batch, rows, cols, channels)` if
        data_format='channels_last'. # Output shape
      4D tensor with shape: `(batch, filters, new_rows, new_cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(batch, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have
        changed due to stride.
    Returns:
      Output tensor of block.
    """
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(
            ((0, 1), (0, 1)), name="conv_pad_%d" % block_id
        )(inputs)
    x = layers.DepthwiseConv2D(
        (3, 3),
        padding="same" if strides == (1, 1) else "valid",
        depth_multiplier=depth_multiplier,
        strides=strides,
        use_bias=False,
        name="conv_dw_%d" % block_id,
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="conv_dw_%d_bn" % block_id
    )(x)
    x = layers.ReLU(6.0, name="conv_dw_%d_relu" % block_id)(x)

    x = layers.Conv2D(
        pointwise_conv_filters,
        (1, 1),
        padding="same",
        use_bias=False,
        strides=(1, 1),
        name="conv_pw_%d" % block_id,
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name="conv_pw_%d_bn" % block_id
    )(x)
    return layers.ReLU(6.0, name="conv_pw_%d_relu" % block_id)(x)

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    """Adds an initial convolution layer (with batch normalization and relu6).
    Args:
      inputs: Input tensor of shape `(rows, cols, 3)` (with `channels_last`
        data format) or (3, rows, cols) (with `channels_first` data format).
        It should have exactly 3 inputs channels, and width and height should
        be no smaller than 32. E.g. `(224, 224, 3)` would be one valid value.
      filters: Integer, the dimensionality of the output space (i.e. the
        number of output filters in the convolution).
      alpha: controls the width of the network. - If `alpha` < 1.0,
        proportionally decreases the number of filters in each layer. - If
        `alpha` > 1.0, proportionally increases the number of filters in each
        layer. - If `alpha` = 1, default number of filters from the paper are
        used at each layer.
      kernel: An integer or tuple/list of 2 integers, specifying the width and
        height of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the convolution along the width and height. Can be a single integer
        to specify the same value for all spatial dimensions. Specifying any
        stride value != 1 is incompatible with specifying any `dilation_rate`
        value != 1. # Input shape
      4D tensor with shape: `(samples, channels, rows, cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, rows, cols, channels)` if
        data_format='channels_last'. # Output shape
      4D tensor with shape: `(samples, filters, new_rows, new_cols)` if
        data_format='channels_first'
      or 4D tensor with shape: `(samples, new_rows, new_cols, filters)` if
        data_format='channels_last'. `rows` and `cols` values might have
        changed due to stride.
    Returns:
      Output tensor of block.
    """
    channel_axis = -1
    filters = int(filters * alpha)
    x = layers.Conv2D(
        filters,
        kernel,
        padding="same",
        use_bias=False,
        strides=strides,
        name="conv1",
    )(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name="conv1_bn")(x)
    return layers.ReLU(6.0, name="conv1_relu")(x)



def MobileNet(
                input_dim=None,
                alpha=1.0,
                depth_multiplier=1,
            ):
    
    inputs = layers.Input(shape=input_dim)
    
    x = _conv_block(inputs, 32, alpha, strides=(2, 2))
    
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    x = layers.GlobalAveragePooling2D()(x)

    return inputs, x




if __name__ == '__main__':
    x, y = MobileNet(input_dim= (120, 257, 1))
    model = tf.keras.models.Model(x, y)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()