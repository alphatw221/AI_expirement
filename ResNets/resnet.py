
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.layers as KE
import tensorflow.keras.utils as KU
import tensorflow.keras.models as KM


def identity_block(input_tensor, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = KL.BatchNormalization()(x)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                  '2c', use_bias=use_bias)(x)
    x = KL.BatchNormalization()(x)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = KL.BatchNormalization()(shortcut)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet101(input_image, train_bn=True):
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    
    for i in range(22): # block_count = {"resnet50": 5, "resnet101": 22}
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
    C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)

    pool = tf.math.reduce_mean(C5, [1, 2])
    x=KL.Dropout(0.2)(pool)
    x=KL.Dense(256,name="srink_dense", activation="relu")(x)
    x=KL.Dense(2,name="output_dense", use_bias=False, activation="softmax")(x)
    return x

def resnet50(input_image, train_bn=True):
    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    
    for i in range(5): # block_count = {"resnet50": 5, "resnet101": 22}
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
    C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)

    pool = tf.math.reduce_mean(C5, [1, 2])
    x=KL.Dropout(0.2)(pool)
    x=KL.Dense(256,name="srink_dense", activation="relu")(x)
    x=KL.Dense(2,name="output_dense", use_bias=False, activation="softmax")(x)
    return x

def r101_subtract(input_image, train_bn=True):
    goldImage = input_image[:,:,:,:3]
    scanImage = input_image[:,:,:,3:]

    conv1 = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(goldImage)
    x = conv1(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    x1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    y = KL.ZeroPadding2D((3, 3))(scanImage)
    y = conv1(y)
    y = KL.BatchNormalization()(y)
    y = KL.Activation('relu')(y)
    y1 = y = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(y)

    # Subtract
    x = KL.Subtract()([x, y])

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    for i in range(22): # block_count = {"resnet50": 5, "resnet101": 22}
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    x4 = x

    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
    C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)

    pool = tf.math.reduce_mean(C5, [1, 2])
    x=KL.Dropout(0.2)(pool)
    x=KL.Dense(256,name="srink_dense", activation="relu")(x)
    x=KL.Dense(2,name="output_dense", use_bias=False, activation="softmax")(x)
    return x

def r50_subtract(input_image, train_bn=True):
    goldImage = input_image[:,:,:,:3]
    scanImage = input_image[:,:,:,3:]

    conv1 = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)

    # Stage 1
    x = KL.ZeroPadding2D((3, 3))(goldImage)
    x = conv1(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)
    x1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    y = KL.ZeroPadding2D((3, 3))(scanImage)
    y = conv1(y)
    y = KL.BatchNormalization()(y)
    y = KL.Activation('relu')(y)
    y1 = y = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(y)

    # Subtract
    x = KL.Subtract()([x, y])

    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)

    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
    for i in range(5): 
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    x4 = x

    # Stage 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
    C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)

    pool = tf.math.reduce_mean(C5, [1, 2])
    x=KL.Dropout(0.2)(pool)
    x=KL.Dense(256,name="srink_dense", activation="relu")(x)
    x=KL.Dense(2,name="output_dense", use_bias=False, activation="softmax")(x)
    return x


def identity_block_share(input_x, input_y, kernel_size, filters, stage, block, use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    conv_2a = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=use_bias)
    conv_2b = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)
    conv_2c = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=use_bias)

    x = conv_2a(input_x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    x = conv_2b(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    x = conv_2c(x)
    x = KL.BatchNormalization()(x)

    x = KL.Add()([x, input_x])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out_x')(x)

    y = conv_2a(input_y)
    y = KL.BatchNormalization()(y)
    y = KL.Activation('relu')(y)

    y = conv_2b(y)
    y = KL.BatchNormalization()(y)
    y = KL.Activation('relu')(y)

    y = conv_2c(y)
    y = KL.BatchNormalization()(y)

    y = KL.Add()([y, input_y])
    y = KL.Activation('relu', name='res' + str(stage) + block + '_out_y')(y)

    return x, y


def conv_block_share(input_x, input_y, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    conv_2a = KL.Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=use_bias)
    conv_2b = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias)
    conv_2c = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +'2c', use_bias=use_bias)
    conv_1 = KL.Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias)

    x = conv_2a(input_x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    x = conv_2b(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu')(x)

    x = conv_2c(x)
    x = KL.BatchNormalization()(x)

    shortcut_x = conv_1(input_x)
    shortcut_x = KL.BatchNormalization()(shortcut_x)

    x = KL.Add()([x, shortcut_x])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out_x')(x)

    y = conv_2a(input_y)
    y = KL.BatchNormalization()(y)
    y = KL.Activation('relu')(y)

    y = conv_2b(y)
    y = KL.BatchNormalization()(y)
    y = KL.Activation('relu')(y)

    y = conv_2c(y)
    y = KL.BatchNormalization()(y)

    shortcut_y = conv_1(input_y)
    shortcut_y = KL.BatchNormalization()(shortcut_y)

    y = KL.Add()([y, shortcut_y])
    y = KL.Activation('relu', name='res' + str(stage) + block + '_out_y')(x)
    return x, y