from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.models import model_from_json

import numpy as np
import pandas as pd
import bcolz
import threading

from time import time
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


class RobocarTrainer(object):
    
    # data for training
    input_shape=(224,224,3)
    im_shape = (224, 224)

    # train parameters
    batch_size = 16
    n_epochs = 100

    def __init__(self, model_name, model_path, dataset_path, log_path='trainlogs'):
        self.model_name = model_name
        self.model_path = model_path
        self.log_path = log_path
        self.dataset_path = dataset_path

        # creating callbacks
        self.tfBoardCB = TensorBoard('{}/{}_{}'.format(self.log_path, model_name, time()), write_graph=True)

        filepath= model_path + model_name + '_best.h5'

        self.checkpointCB = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    
    def LoadDataset(self):

        print('loading dataset...')
        self.dataset = pd.read_csv(self.dataset_path + 'target.csv')

        self.dataset['imgpath'] = self.dataset.id.apply(file_path_from_db_id, args=("%d.bmp", self.dataset_path))

        self.train, self.test = train_test_split(self.dataset, test_size=0.2)
        self.valid, self.test = train_test_split(self.test, test_size=0.7)
    
        self.train_steps = int(self.train.shape[0] / self.batch_size)
        self.valid_steps = int(self.valid.shape[0] / self.batch_size)
        self.test_steps = int(self.test.shape[0] / self.batch_size)
        print('dataset loaded!')

    def Train(self):

        print('loading model...')
        self.model = models.simple1(self.input_shape)
        self.model.summary()
        model_json = self.model.to_json()
        with open(self.model_path + self.model_name + '.json', "w") as json_file:
            json_file.write(model_json)
            
        plot_model(self.model, to_file=self.model_name + '.png', show_shapes=True)
        print('dataset size: ', self.train.shape[0],
                'train_steps: ', self.train_steps, 
                'valid steps: ', self.valid_steps, 
                'test_steps: ', self.test_steps)
        
        print('hiperparameters:')
        print('batch size:{}'.format(self.batch_size))
        

        print('training...')
        self.model.fit_generator(
                            generator_from_df(self.train, self.batch_size, self.im_shape, 'angular'),
                            steps_per_epoch=self.train_steps, 
                            epochs=self.n_epochs,
                            validation_data=generator_from_df(self.valid, self.batch_size, self.im_shape, 'angular'),
                            validation_steps=self.valid_steps,
                            callbacks=[self.tfBoardCB, self.checkpointCB],
                            verbose=2
                            )
        print('finished training')
        self.score = self.model.evaluate_generator(
                            generator_from_df(self.test, self.batch_size, self.im_shape, 'angular'), 
                            steps=self.test_steps
                            )
        print('loss: ', self.score)

    def SaveModel(self):
        # serialize model to JSON
        self.model_json = self.model.to_json()
        with open(self.model_name + '.json', "w") as json_file:
            json_file.write(self.model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.model_name + '.h5')
        print("Saved model to disk")




if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='Entrenamiento de una red neuronal convolucional')
    parser.add_argument("model_name", help="name of the model to train")
    parser.add_argument("model_path", help="path where the model files will be saved")
    parser.add_argument("dataset_path", help="path where the the dataset is saved")
    
    args = parser.parse_args()
    
    trainer = RobocarTrainer(args.model_name, args.model_path, args.dataset_path)
    trainer.LoadDataset()
    trainer.Train()
    trainer.SaveModel()