from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.models import model_from_json

import numpy as np
import pandas as pd
import bcolz
import threading

import os
import sys
import glob
import shutil

from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.utils import plot_model
import models
from utils import *

if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser(description='reentrena un modelo de keras')
    parser.add_argument('-rpi', action='store_true',help='for raspberry pi hardware (default: false)')
    parser.add_argument('-model_name', action='store', type=str, help='keras model name')
    parser.add_argument('-epochs', action='store', type=int, help='keras model name')
    
    
    args = parser.parse_args()
    print args.rpi

    
    model_name = 'white1'

    if args.model_name != None:
        model_name = args.model_name

    # load json and create model
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(model_name + ".h5")
    print("Loaded model from disk")
    
    

    loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    ## callbacks

    tfBoardCB = TensorBoard('trainlogs', histogram_freq=1, write_graph=True, write_images=False)
    filepath='trainlogs/' + model_name + '_best_retrained.h5'
    checkpointCB = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    ########################

    dataset = pd.read_csv('dataset/target.csv')

    dataset['imgpath'] = dataset.id.apply(file_path_from_db_id)

    train, test = train_test_split(dataset, test_size=0.2)
    valid, test = train_test_split(test, test_size=0.7)

    train['target'] = train['target'].apply(extract)
    test['target'] = test['target'].apply(extract)

    input_shape=(224,224,3)
    model = models.conv2(input_shape)

    model.summary()
    plot_model(model, to_file=model_name + '.png')

    ## evaluate before retraining
    # evaluate loaded model on test data

    score1 = loaded_model.evaluate_generator(
                        generator_from_df(test, batch_size, (224,224), 'angular'), 
                        steps=test_steps
                        )
    print('score 1: ', score1)
    ### train parameters

    batch_size = 8
    train_steps = int(train.shape[0] / batch_size)
    valid_steps = int(valid.shape[0] / batch_size)
    test_steps = int(test.shape[0] / batch_size)

    n_epochs = 70
    if args.epochs != None:
        n_epochs = args.epochs

    print('dataset size: ', train.shape[0],'train_steps: ', train_steps, 'valid steps: ', valid_steps, 'test_steps: ', test_steps)
    loaded_model.fit_generator(
                        generator_from_df(train, batch_size, (224,224), 'angular'),
                        steps_per_epoch=train_steps, 
                        epochs=n_epochs,
                        validation_data=generator_from_df(valid, batch_size, (224,224), 'angular'),
                        validation_steps=valid_steps,
                        callbacks=[tfBoardCB, checkpointCB]
                        )

    score2 = loaded_model.evaluate_generator(
                        generator_from_df(test, batch_size, (224,224), 'angular'), 
                        steps=test_steps
                        )

    print('=======================================')
    print('score1: ', score1, ' score2: ', score2)
    print('=======================================')
    
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_name + '.h5')
    print("Saved model to disk")