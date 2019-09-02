import tensorflow as tf
import tensorflow.keras as keras
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
    # batch_size = 16
 

    def __init__(self, model_type='base', 
                        model_name='test', 
                        model_path='models', 
                        dataset_path='dataset', 
                        log_path='trainlogs', 
                        verbose=0, 
                        n_epochs=10,
                        batch_size=16,
                        use_generator=True):
        self.model_type = model_type
        self.model_name = model_name
        self.model_path = model_path
        self.log_path = log_path
        self.dataset_path = dataset_path
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_generator = use_generator
        # creating callbacks
        self.tfBoardCB = TensorBoard('{}/{}_{}'.format(self.log_path, model_name, time()), write_graph=True)
        filepath= os.path.join(self.model_path, self.model_name + '_best.h5')
        self.checkpointCB = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        # load model
        print('loading model...')
        print('Using Model Type: {}'.format(self.model_type))
        self.model = getattr(models, self.model_type)(self.input_shape)
        if self.verbose > 0:
            self.model.summary()

    
    def LoadDataset(self):
        print('loading dataset from: {}'.format(self.dataset_path))
        if not os.path.isdir(self.dataset_path): 
            print('[ERROR] No existe la carpeta')
            return False
        
        if not os.path.isfile(os.path.join(self.dataset_path, 'target.csv')):
            print('[ERROR] No existe el archivo target.csv')
            return False
        print('leyendo dataframe...')
        self.dataset = pd.read_csv(os.path.join(self.dataset_path, 'target.csv'))
        self.dataset['imgpath'] = self.dataset.id.apply(file_path_from_db_id, args=("%d.bmp", self.dataset_path))
        print('dividiendo datasets')
        self.train, self.test = train_test_split(self.dataset, test_size=0.2)
        self.valid, self.test = train_test_split(self.test, test_size=0.7)
        self.train_steps = int(self.train.shape[0] / self.batch_size)
        self.valid_steps = int(self.valid.shape[0] / self.batch_size)
        self.test_steps = int(self.test.shape[0] / self.batch_size)
        if not self.use_generator:
            print('load images to memory')
            self.X_train, self.Y_train = dataset_from_df(self.train,self.im_shape, 'angular')
            self.X_valid, self.Y_valid = dataset_from_df(self.valid,self.im_shape, 'angular')
            self.X_test, self.Y_test = dataset_from_df(self.test,self.im_shape, 'angular')
            
            print('Trainset Dimensions: {}'.format(self.X_train.shape))
            print('validset Dimensions: {}'.format(self.X_valid.shape))
            print('Testset Dimensions: {}'.format(self.X_test.shape))

        print('dataset loaded!')
        return True

    def Train(self):
        model_json = self.model.to_json()
        with open(os.path.join(self.model_path, self.model_name + '.json'), "w") as json_file:
            json_file.write(model_json)
            
        plot_model(self.model, to_file=self.model_name + '.png', show_shapes=True)
        print('dataset size: ', self.train.shape[0],
                'train_steps: ', self.train_steps, 
                'valid steps: ', self.valid_steps, 
                'test_steps: ', self.test_steps)
        
        print('hiperparameters:')
        print('batch size:{}'.format(self.batch_size))
        
        start_time = time.time()
        if self.use_generator:
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
        else:
            # load dataset on memory
            print('training begins')
            self.model.fit(self.X_train, self.Y_train,
                            batch_size=self.batch_size,
                            epochs=self.n_epochs,
                            validation_data=(self.X_valid, self.Y_valid),
                            callbacks=[self.tfBoardCB, self.checkpointCB],
                            verbose=2
                            )
            print('finished training')
            self.score = self.model.evaluate_generator(
                                self.X_test, self.Y_test, 
                                batch_size=self.batch_size,
                                )
            # pass
        print('Entrenamiento Finalizado, {} epocas en {0:.2f} [s]'.format(self.n_epochs, time.time() - start_time))
        print('loss: ', self.score)

    def SaveModel(self):
        # serialize model to JSON
        filepath= os.path.join(self.model_path, self.model_name + '.h5')
        self.model.save_weights(filepath)
        print("Saved model to disk")




if __name__ == '__main__':
    
    import argparse
    ap = argparse.ArgumentParser(description='Entrenamiento de una red neuronal convolucional')
    ap.add_argument("-t", "--type", type=str, default='base', help="type of the model to use")
    ap.add_argument("-n", "--name", type=str, default='test', help="name of the trained model")
    ap.add_argument("-p", "--path", type=str, default='models', help="path where the model files will be saved")
    ap.add_argument("-d", "--dataset", type=str, default='dataset/', help="path where the the dataset is saved")
    
    # ap.add_argument("-i", "--input", type=str,help="path to optional input video file")
    # ap.add_argument("-c", "--confidence", type=float, default=0.4,help="minimum probability to filter weak detections")
    # ap.add_argument("-s", "--skip-frames", type=int, default=30,help="# of skip frames between detections")
    args = vars(ap.parse_args())

    trainer = RobocarTrainer(model_type=args['type'], model_name=args['name'], model_path=args['path'], dataset_path=args['dataset'])
    if not trainer.LoadDataset():
        print('No se pudo cargar dataset!')
    else:
        trainer.Train()
        trainer.SaveModel()