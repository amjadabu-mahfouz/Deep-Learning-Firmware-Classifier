# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 13:39:37 2021

@author: user
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir
from os.path import isfile, join
import os.path as pth
import ImageConverter

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


datagen = ImageDataGenerator(
        height_shift_range=0.5,
        rescale=1./255,
        fill_mode='wrap')


#set the path variables and run
path = './dataset/buffalo_vulnerable'
path2 = './dataset/buffalo_vulnerable_aug'

files = [file for file in listdir(path) if isfile(join(path, file))]

for file in files:
    print('file name: ' + file)
    img = load_img(path + '/' + file)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir= path2, save_prefix=file, save_format='png'):
        i += 1
        if i > 4:
            break  # otherwise the generator would loop indefinitely


#dont forget these paths too!
path = './dataset/buffalo_benign'
path2 = './dataset/buffalo_benign_aug'

files = [file for file in listdir(path) if isfile(join(path, file))]

for file in files:
    print('file name: ' + file)
    img = load_img(path + '/' + file)
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape)
    
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir= path2, save_prefix=file, save_format='png'):
        i += 1
        if i > 4:
            break 
