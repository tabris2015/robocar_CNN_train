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

model_name = 'simple22'


## callbacks

tfBoardCB = TensorBoard('trainlogs/{}_{}'.format(model_name, time()), write_graph=True)

filepath='best_models/' + model_name + '_best.h5'
checkpointCB = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
########################

dataset = pd.read_csv('dataset/target.csv')

dataset['imgpath'] = dataset.id.apply(file_path_from_db_id)

train, test = train_test_split(dataset, test_size=0.2)
valid, test = train_test_split(test, test_size=0.7)


input_shape=(224,224,3)

im_shape = (224, 224)

model = models.simple2(input_shape)

model.summary()
plot_model(model, to_file=model_name + '.png', show_shapes=True)
### train parameters

batch_size = 10
train_steps = int(train.shape[0] / batch_size)
valid_steps = int(valid.shape[0] / batch_size)
test_steps = int(test.shape[0] / batch_size)

n_epochs = 100

print('dataset size: ', train.shape[0],'train_steps: ', train_steps, 'valid steps: ', valid_steps, 'test_steps: ', test_steps)
model.fit_generator(
                    generator_from_df(train, batch_size, im_shape, 'angular'),
                    steps_per_epoch=train_steps, 
                    epochs=n_epochs,
                    validation_data=generator_from_df(valid, batch_size, im_shape, 'angular'),
                    validation_steps=valid_steps,
                    callbacks=[tfBoardCB, checkpointCB]
                    )

score = model.evaluate_generator(
                    generator_from_df(test, batch_size, im_shape, 'angular'), 
                    steps=test_steps
                    )

print('loss: ', score)

# serialize model to JSON
model_json = model.to_json()
with open(model_name + '.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(model_name + '.h5')
print("Saved model to disk")