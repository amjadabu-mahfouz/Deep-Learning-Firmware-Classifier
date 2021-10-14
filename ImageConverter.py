# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:34:47 2021

@author: user
"""

import numpy as np

from math import sqrt, ceil

from PIL import Image


class ImageConverter:
    def __init__(self, file_name, image_type, image_name):
        self.image_type = image_type
        
        # this will take user input for the file location
        self.file_to_convert = file_name
      
        
        # the with statement wraps it's code in "__enter__()" and "__exit__()" statements much like try and catch statements 
        # this will open the file and read it's bytes into the "data" variable
        with open(self.file_to_convert, 'rb') as my_file:
            data = my_file.read()
            
        # this will get the length of the file in bytes 
        data_length = len(data)
        
            
        
        # this will convert the buffer into a 1D array; dtype refers to the datatype of the returned array; uint8 is an integer datatype from 0-255 that represents bytes
        d = np.frombuffer(data, dtype=np.uint8)
        
        
        # the 1D array has to be processed into a 2D one and resized to a square so that it could be saved as an impage
        
        # 1) the first step is to take the square root of the file size (in bytes) and round up  
        sqrt_length = int( ceil( sqrt( data_length)))
        # 2) use the square root length from step 1 and compute the size of theconverted image 
        image_length = sqrt_length * sqrt_length
        # 3) compute the padding required's length for the converted image 
        padding_length = image_length - data_length
        
        # the hstack method is used to combine or "stack" arrays together, in this case the padding; np.zeros will return an array of zeros in the specified length and format
        data_final_image = np.hstack((d, np.zeros(padding_length, np.uint8)))
        
        # the np.reshape method reshapoes an array without affecting its data; the second parameter is for number of rows and columns
        final_image = np.reshape(data_final_image, (sqrt_length, sqrt_length))
        
        if int(image_type) == 0:
            grayscale = Image.fromarray(final_image)
            grayscale.save(image_name + ".png")
            
        elif int(image_type) == 1: 
            RGB = Image.fromarray(final_image.astype(int), 'RGB')
            RGB.save(image_name + ".png")
            
        
        