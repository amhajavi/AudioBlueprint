# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow.keras.layers as layers
import tensorflow.keras.backend as backend
from keras.applications import imagenet_utils


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    """Inverted ResNet block."""
    channel_axis = -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    # Ensure the number of filters on the last 1x1 convolution is divisible by
    # 8.
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = "block_{}_".format(block_id)

    if block_id:
        # Expand with a pointwise 1x1 convolution.
        x = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding="same",
            use_bias=False,
            activation=None,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand_BN",
        )(x)
        x = layers.ReLU(6.0, name=prefix + "expand_relu")(x)
    else:
        prefix = "expanded_conv_"

    # Depthwise 3x3 convolution.
    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, 3), name=prefix + "pad"
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding="same" if stride == 1 else "valid",
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise_BN",
    )(x)

    x = layers.ReLU(6.0, name=prefix + "depthwise_relu")(x)

    # Project with a pointwise 1x1 convolution.
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        activation=None,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project_BN",
    )(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + "add")([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



def MobileNetV2(
	     			input_dim=None,
	                alpha=1.0,
	                depth_multiplier=1,
	            ):
    
    inputs = layers.Input(shape=input_dim)
   
    channel_axis = -1 

    first_block_filters = _make_divisible(32 * alpha, 8)
    
    x = layers.Conv2D(
				        first_block_filters,
				        kernel_size=3,
				        strides=(2, 2),
				        padding="same",
				        use_bias=False,
				        name="Conv1",
    				  )(inputs)
    
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name="bn_Conv1")(x)
    
    x = layers.ReLU(6.0, name="Conv1_relu")(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
    
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
    
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
    
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
    
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
    
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
    
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
    
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
    
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
    
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
    
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we increase the number of output
    # channels.
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters, kernel_size=1, use_bias=False, name="Conv_1")(x)
    
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1_bn")(x)
    
    x = layers.ReLU(6.0, name="out_relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    return inputs, x
