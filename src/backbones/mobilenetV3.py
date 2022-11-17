# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])

def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(
        keepdims=True, name=prefix + "squeeze_excite/AvgPool"
    )(inputs)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv",
    )(x)
    x = layers.ReLU(name=prefix + "squeeze_excite/Relu")(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        name=prefix + "squeeze_excite/Conv_1",
    )(x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + "squeeze_excite/Mul")([inputs, x])
    return x


def _inverted_res_block(
    x, expansion, filters, kernel_size, stride, se_ratio, activation, block_id
):
    channel_axis = -1
    shortcut = x
    prefix = "expanded_conv/"
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = "expanded_conv_{}/".format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "expand/BatchNorm",
        )(x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=prefix + "depthwise/pad",
        )(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding="same" if stride == 1 else "valid",
        use_bias=False,
        name=prefix + "depthwise",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "depthwise/BatchNorm",
    )(x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project",
    )(x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + "project/BatchNorm",
    )(x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + "Add")([shortcut, x])
    return x


def MobileNetV3Small(
                    input_dim=None,
                    last_point_ch=1024,
                    alpha=1.0,
                    minimalistic=False,
                    dropout_rate=0.2,
                ):
    
    inputs = layers.Input(shape=input_dim)

    channel_axis = -1

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25


    x = layers.Conv2D(
                        16,
                        kernel_size=3,
                        strides=(2, 2),
                        padding="same",
                        use_bias=False,
                        name="Conv"
                    )(inputs)
    
    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv/BatchNorm")(x)

    x = activation(x)

    def depth(d):
            return _depth(d * alpha)

    x = _inverted_res_block(x, 1, depth(16), 3, 2, se_ratio, relu, 0)
    
    x = _inverted_res_block(x, 72.0 / 16, depth(24), 3, 2, None, relu, 1)
    
    x = _inverted_res_block(x, 88.0 / 24, depth(24), 3, 1, None, relu, 2)
    
    x = _inverted_res_block(x, 4, depth(40), kernel, 2, se_ratio, activation, 3)
    
    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 4)

    x = _inverted_res_block(x, 6, depth(40), kernel, 1, se_ratio, activation, 5)

    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 6)
    
    x = _inverted_res_block(x, 3, depth(48), kernel, 1, se_ratio, activation, 7)

    x = _inverted_res_block(x, 6, depth(96), kernel, 2, se_ratio, activation, 8)
    
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 9)
    
    x = _inverted_res_block(x, 6, depth(96), kernel, 1, se_ratio, activation, 10)
    
    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_point_ch = _depth(last_point_ch * alpha)
    
    x = layers.Conv2D(
                        last_conv_ch,
                        kernel_size=1,
                        padding="same",
                        use_bias=False,
                        name="Conv_1"
                      )(x)

    x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999, name="Conv_1/BatchNorm")(x)
    
    x = activation(x)

    x = layers.GlobalAveragePooling2D()(x)

    return inputs, x
