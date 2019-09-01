import tensorflow as tf
import tensorflow.keras as keras
from keras.models import model_from_json
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation
from keras.layers import ELU, Lambda, merge, GlobalAveragePooling2D

from keras.optimizers import Adam

from keras import backend as K

def base(input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def conv1(input_shape):
    model = Sequential()
    model.add(Conv2D(3, kernel_size=(1, 1), activation='relu', input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def conv2(input_shape):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(5, 5), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='tanh'))


    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def custom_loss(y_true, y_pred):
    
    loss = tf.square(y_true - y_pred)
    loss = .5 * tf.reduce_mean(loss)
    return loss

def simple1(input_shape):   
    # this network is used with a 80 x 160 image size 
    # Construct the network 
    image_inp = Input(shape=input_shape)

    x = Conv2D(filters=16, kernel_size=(3, 5), activation='relu', padding='valid')(image_inp)
    x = Conv2D(filters=16, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=32, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = Conv2D(filters=32, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=4,  kernel_size=(1, 1), activation='linear', padding='same')(x)

    x = Flatten()(x)

    x = Dense(1, activation='tanh', kernel_regularizer='l1')(x)

    angle_out = x

    model = Model(inputs=[image_inp], outputs=[angle_out])
    model.compile(loss=custom_loss, optimizer='adam')

    return model

def simple2(input_shape):   
    # this network is used with a 80 x 160 image size 
    # Construct the network 
    image_inp = Input(shape=input_shape)

    x = Conv2D(filters=16, kernel_size=(3, 5), activation='relu', padding='valid')(image_inp)
    x = Conv2D(filters=16, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=32, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = Conv2D(filters=32, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = Conv2D(filters=64, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=4,  kernel_size=(1, 1), activation='linear', padding='same')(x)
    
    x = Flatten()(x)

    x = Dense(1, activation='tanh', kernel_regularizer='l1')(x)

    angle_out = x

    model = Model(inputs=[image_inp], outputs=[angle_out])
    model.compile(loss=custom_loss, optimizer='adam')

    return model


def simple3(input_shape):   
    # this network is used with a 80 x 160 image size 
    # Construct the network 
    image_inp = Input(shape=input_shape)

    x = Conv2D(filters=4, kernel_size=(1, 1), activation='relu', padding='valid')(image_inp)
    x = Conv2D(filters=16, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=32, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = Conv2D(filters=64, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = Conv2D(filters=128, kernel_size=(3, 5), activation='relu', padding='valid')(x)
    x = MaxPooling2D((4, 2))(x)

    x = Conv2D(filters=4,  kernel_size=(1, 1), activation='linear', padding='same')(x)
    
    x = Flatten()(x)

    x = Dense(1, activation='tanh', kernel_regularizer='l1')(x)

    angle_out = x

    model = Model(inputs=[image_inp], outputs=[angle_out])
    model.compile(loss='mse', optimizer='adam')

    return model

def fire_module(x, fire_id, squeeze=32, expand=64):
    """
    This is a modified version of: https://github.com/rcmalli/keras-squeezenet/blob/master/squeezenet.py#L14
    Changes made:
    * Uses ELU activation
    * Only supports tf
    """
    s_id = 'fire' + str(fire_id) + '/'
    c_axis = 3
    sq1x1 = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    elu = "elu_"

    x = Conv2D(squeeze, 1, 1, border_mode='valid', name=s_id + sq1x1)(x)
    x = Activation('elu', name=s_id + elu + sq1x1)(x)

    left = Conv2D(expand, 1, 1, border_mode='valid', name=s_id + exp1x1)(x)
    left = Activation('elu', name=s_id + elu + exp1x1)(left)

    right = Conv2D(expand, 3, 3, border_mode='same', name=s_id + exp3x3)(x)
    right = Activation('elu', name=s_id + elu + exp3x3)(right)

    x = merge([left, right], mode='concat', concat_axis=c_axis, name=s_id + 'concat')
    return x

def squeeze_model_52(input_shape):
    """
    This model is a modification from the reference:
    https://github.com/rcmalli/keras-squeezenet/blob/master/squeezenet.py
    Normalizing will be done in the model directly for GPU speedup
    """
    input_img = Input(shape=input_shape)
    
    x = Conv2D(4, 3, 3, subsample=(2, 2), border_mode='valid', name='conv1')(input_img)
    x = Activation('elu', name='elu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=1, expand=2)
    x = Dropout(0.2, name='drop3')(x)


    x = GlobalAveragePooling2D()(x)
    out = Dense(1, name='loss')(x)
    model = Model(input=input_img, output=[out])

    model.compile(optimizer=Adam(lr=0.06), loss='mse')
    return model
