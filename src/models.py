import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def lstm(input_dim=(10, 12, 257, 1), num_class=12):

    from backbones.lstm import lstm

    inputs, x = lstm(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)

    model = keras.models.Model(inputs, y)

    return model


def gru(input_dim=(10, 12, 257, 1), num_class=12):

    from backbones.gru import gru

    inputs, x = gru(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)

    model = keras.models.Model(inputs, y)

    return model


def crnn(input_dim=(10, 12, 257, 1), num_class=12):

    from backbones.crnn import crnn

    inputs, x = crnn(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)

    model = keras.models.Model(inputs, y)

    return model


def mobilenet(input_dim=(10, 12, 257, 1), num_class=12):

    from backbones.mobilenet import MobileNet

    inputs, x = MobileNet(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)

    model = keras.models.Model(inputs, y)

    return model


def mobilenetV2(input_dim=(10, 12, 257, 1), num_class=12):

    from backbones.mobilenetV2 import MobileNetV2

    inputs, x = MobileNetV2(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)

    model = keras.models.Model(inputs, y)

    return model


def mobilenetV3small(input_dim=(10, 12, 257, 1), num_class=12):

    from backbones.mobilenetV3 import MobileNetV3Small

    inputs, x = MobileNetV3Small(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)

    model = keras.models.Model(inputs, y)

    return model



def resnet(input_dim=(120, 257, 1), num_class=12):

    from backbones.thin_resnet import thin_resnet

    inputs, x = thin_resnet(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)

    model = keras.models.Model(inputs, y)

    return model


def se_resnet(input_dim=(120, 257, 1), num_class=12):

    from backbones.se_resnet import thin_seresnet

    inputs, x = thin_seresnet(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)
    model = keras.models.Model(inputs, y)

    return model


def vgg_M(input_dim=(257, 250, 1), num_class=12):

    from backbones.vgg import VGG_M

    inputs, x = VGG_M(input_dim=input_dim)

    y = keras.layers.Dense(num_class,
                           activation='softmax',
                           kernel_initializer='orthogonal',
                           use_bias=False,
                           kernel_regularizer=keras.regularizers.l2(1e-4),
                           bias_regularizer=keras.regularizers.l2(1e-4),
                           name='classification')(x)
    model = keras.models.Model(inputs, y)

    return model


def dummy(x, y):
    print("function_works")
    return x


def create_model(args, input_shape, number_of_classes):
    # check compatibility of models and train_set
    if args.sequential_model:
        if args.model not in ['LSTM', 'GRU', 'CRNN']:
            raise ValueError('{} is not compatible with --sequential_model. Please Select from {}'.format(args.model, ['LSTM', 'GRU', 'CRNN']))
    elif args.model not in ['MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'ResNet', 'SEResNet', 'VGG_M']:
        raise ValueError('{} is not compatible with non sequential execution. Please Select from {}'.format(args.model, ['MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'ResNet', 'SEResNet', 'VGG_M']))

    match args.model:
        case 'LSTM':
            model = lstm(input_shape, number_of_classes)
        case 'GRU':
            model = gru(input_shape, number_of_classes)
        case 'CRNN':
            model = crnn(input_shape, number_of_classes)
        case 'MobileNet':
            model = mobilenet(input_shape, number_of_classes)
        case 'MobileNetV2':
            model = mobilenetV2(input_shape, number_of_classes)
        case 'MobileNetV3Small':
            model = mobilenetV3small(input_shape, number_of_classes)
        case 'ResNet':
            model = resnet(input_shape, number_of_classes)
        case 'SEResNet' :
            model = se_resnet(input_shape, number_of_classes)
        case 'VGG_M':
            model = vgg_M(input_shape, number_of_classes)
        case _:
            raise ValueError('Please Select the model from {}'.format(args.model, ['LSTM', 'GRU', 'CRNN', 'MobileNet', 'MobileNetV2', 'MobileNetV3Small', 'ResNet', 'SEResNet', 'VGG_M']))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model
