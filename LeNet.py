import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D
from keras import callbacks
from keras.datasets import mnist
from keras.utils import np_utils



def LeNet(input_tensor=None, input_shape=(32, 32, 1), activation='relu'):
    
    act = activation
     
    if input_tensor is None:
        input_tensor = Input(shape = input_shape)
    x = ZeroPadding2D(padding=(2,2), data_format='channels_last')(input_tensor)
    x = Conv2D(filters=6, kernel_size=5, strides=1, activation=act)(x)
    x = MaxPooling2D(pool_size=(2,2), strides=2)(x)
    x = Conv2D(filters=16, kernel_size=5, strides=1, activation=act)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = Flatten()(x)
    x = Dense(units=120, activation=act)(x)
    x = Dense(units=84, activation=act)(x)
    x = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)

    return model