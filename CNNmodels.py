# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 22:17:28 2021

@author: user
"""

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator  

from tensorflow.keras.layers import BatchNormalization
from keras.layers import Dropout

from keras import regularizers
from keras.optimizers import RMSprop
from keras.optimizers import SGD

class CNNmodels:
    def __init__ (self):
        self.opt = SGD(lr=0.001)
        # kernel_regularizer = regularizers.l2( l=0.01)
        #loss = 'kullback_leibler_divergence'
        
        
    def make_Models(self):
        opt = self.opt
        # this is Model-1 (1 conv layer)
        cnn1 = tf.keras.models.Sequential()  
        
        cnn1.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu', input_shape=[500, 500, 1])) 
        cnn1.add(BatchNormalization()) 
        cnn1.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        
        cnn1.add(tf.keras.layers.Flatten()) 
        
        cnn1.add(tf.keras.layers.Dense(units=128, activation='relu'))
        
        cnn1.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        cnn1.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])   
        
        cnn1.save('./CNN models/Model-1.h5')
        
        
        
        # this is Model-2 with 2 conv back to back layers with 1 pooling
        
        cnn2 = tf.keras.models.Sequential()  
        
        cnn2.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu', input_shape=[500, 500, 1])) 
        cnn2.add(tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu')) 
        cnn2.add(BatchNormalization()) 
        cnn2.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        
        
        cnn2.add(tf.keras.layers.Flatten()) 
        
        cnn2.add(tf.keras.layers.Dense(units=128, activation='relu'))
        
        
        
        cnn2.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        cnn2.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])   
        
        cnn2.save('./CNN models/Model-2.h5')
        
        
        
        # this is Model-3 with 2 conv (seperate) and 2 pooling layers 
        
        cnn3 = tf.keras.models.Sequential()  
        
        cnn3.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu', input_shape=[500, 500, 1])) 
        cnn3.add(BatchNormalization()) 
        cnn3.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        
        cnn3.add(tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu')) 
        cnn3.add(BatchNormalization()) 
        cnn3.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        
        cnn3.add(tf.keras.layers.Flatten()) 
        
        cnn3.add(tf.keras.layers.Dense(units=128, activation='relu'))
        cnn3.add(tf.keras.layers.Dense(units=256, activation='relu'))
        
        cnn3.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        cnn3.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])   
        
        cnn3.save('./CNN models/Model-3.h5')
        
        
        # this is Model-4 with 3 conv (seperate) and 3 pooling layers 
        
        cnn4 = tf.keras.models.Sequential()  
        
        cnn4.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu', input_shape=[500, 500, 1])) 
        cnn4.add(BatchNormalization()) 
        cnn4.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        cnn4.add(tf.keras.layers.Conv2D(filters=10, kernel_size=3, activation='relu')) 
        cnn4.add(BatchNormalization()) 
        cnn4.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        cnn4.add(tf.keras.layers.Conv2D(filters=20, kernel_size=3, activation='relu')) 
        cnn4.add(BatchNormalization()) 
        cnn4.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
        
        cnn4.add(tf.keras.layers.Flatten()) 
        
        cnn4.add(tf.keras.layers.Dense(units=128, activation='relu'))
        
        cnn4.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        cnn4.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])   
        
        cnn4.save('./CNN models/Model-4.h5')
        
        
        
        
        
        # this is Model-5 with 2 desnse layers 
        
        cnn5 = tf.keras.models.Sequential()  
        
        
        cnn5.add(tf.keras.layers.Flatten(input_shape=(500, 500,1))) 
        
        cnn5.add(tf.keras.layers.Dense(units=128, activation='relu'))
        
        cnn5.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        cnn5.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])   
        
        cnn5.save('./CNN models/Model-5.h5')
        
        
        
        # this is Model-6 with 2 desnse layers 
        
        cnn6 = tf.keras.models.Sequential()  
        
        cnn6.add(tf.keras.layers.Flatten(input_shape=(500, 500,1))) 
        
        cnn6.add(tf.keras.layers.Dense(units=128, activation='relu'))
        
        cnn6.add(tf.keras.layers.Dense(units=256, activation='relu'))
        
        cnn6.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        cnn6.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])   
        
        cnn6.save('./CNN models/Model-6.h5')