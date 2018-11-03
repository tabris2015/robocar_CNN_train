from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from keras.models import model_from_json

import numpy as np
import pandas as pd
import bcolz
import threading

import cv2

import os
import sys
import glob
import shutil

from sklearn.model_selection import train_test_split

import models

## data augmentation functions
def horizontal_flip(img, target):
    '''
    Randomly flip image along horizontal axis: 1/2 chance that the image will be flipped
    img: original image in array type
    target: steering angle value of the original image
    '''
    choice = np.random.choice([0,1])
    if choice == 1:
        img, target = cv2.flip(img, 1), -target
    
    return (img, target)


def transf_brightness(img, target):
    '''
    Adjust the brightness of the image, by a randomly generated factor between 0.1 (dark) and 1. (unchanged)
    img: original image in array type
    target: steering angle value of the original image
    '''
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #change Value/Brightness/Luminance: alpha * V
    alpha = np.random.uniform(low=0.15, high=1.0, size=None)
    v = hsv[:,:,2]
    v = v * alpha
    hsv[:,:,2] = v.astype('uint8')
    #min_val = np.min(hsv[:,:,2])
    #max_val = np.max(hsv[:,:,2])
    #print('min:{} | max: {}'.format(min_val, max_val))
    rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
    
    return (rgb, target)


  

def filter_zero_steering(target, del_rate):
    '''
    Randomly pick examples with steering angle of 0, and return their index
    target: list of steering angle value in the original dataset
    del_rate: rate of deletion - del_rate=0.9 means delete 90% of the example with steering angle=0
    '''
    steering_zero_idx = np.where(target == 0)
    steering_zero_idx = steering_zero_idx[0]
    size_del = int( len(steering_zero_idx) * del_rate )
    
    return np.random.choice(steering_zero_idx, size=size_del, replace=False)


def image_transformation(img_adress, target, data_dir):
    # Image swap at random: left-right-center
    img_adress, target = center_RightLeft_swap(img_adress, target )
    # Read img file and convert to RGB
    img = cv2.imread(data_dir + img_adress)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # change luminance: 50% chance
    img, target = transf_brightness(img, target)
    #flip image: 50% chance
    img, target = horizontal_flip(img, target)

### utils

def file_path_from_db_id(db_id, pattern="%d.bmp", top="dataset/"):
    s = '%09d' % db_id
    return os.path.join(top, pattern % db_id)


def generator_from_df(df, batch_size, target_size, target_column='target', features=None, process=True):
    print('generating minibatch!')
    nbatches, n_skipped_per_epoch = divmod(df.shape[0], batch_size)
    #print nbatches
    count = 1
    epoch = 0
    # New epoch.
    while 1:
        df = df.sample(frac=1) # shuffle in every epoch
        epoch += 1
        i, j = 0, batch_size
        # Mini-batches within epoch.
        mini_batches_completed = 0
        for _ in range(nbatches):
            sub = df.iloc[i:j]
            try:
                if process == True:
                    X = np.array([(2 * (img_to_array(load_img(f, target_size=target_size)) / 255.0 - 0.5)) for f in sub.imgpath])
                else:
                    X = np.array([((img_to_array(load_img(f, target_size=target_size)))) for f in sub.imgpath])
                    
                Y = sub[target_column].values
                # Simple model, one input, one output.
                mini_batches_completed += 1
                print ".",
                
                yield X, Y

            except IOError as err:
                count -= 1

            i = j
            j += batch_size
            count += 1

def extract(target):
    return np.array([float(x) for x in (target.replace("[","").replace("]","").split(","))])

