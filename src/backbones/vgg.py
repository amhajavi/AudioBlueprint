from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization, Input
from tensorflow.keras.regularizers import l2

weight_decay = 1e-4


def VGG_M(input_dim):
    
    inputs = Input(shape=input_dim)
    
    x = Conv2D(96, (7, 7),
                strides=2,
                kernel_initializer='orthogonal',
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                activation='relu')(inputs)

    x = BatchNormalization(axis=-1)(x)

    x = MaxPooling2D((3,3), strides=2)(x)


    x = Conv2D(256, (5, 5),
                strides=2,
                kernel_initializer='orthogonal',
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                activation='relu')(x)

    x = BatchNormalization(axis=-1)(x)

    x = MaxPooling2D((3,3), strides=2)(x)

    x = Conv2D(512, (3, 3),
                strides=1,
                kernel_initializer='orthogonal',
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                activation='relu')(x)

    x = Conv2D(512, (3, 3),
                strides=1,
                kernel_initializer='orthogonal',
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                activation='relu')(x)

    x = Conv2D(512, (3, 3),
                strides=1,
                kernel_initializer='orthogonal',
                use_bias=False,
                kernel_regularizer=l2(weight_decay),
                padding='same',
                activation='relu')(x)

    x = GlobalMaxPooling2D()(x)

    return inputs, x

if __name__ == '__main__':
    import tensorflow as tf
    x, y = VGG_M(input_dim= (120, 257, 1))
    model = tf.keras.models.Model(x, y)
    model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()