# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 22:18:41 2021

@author: user
"""

from os import listdir
from os.path import isfile, join
import ImageConverter
import Image_Feature_Extraction




#path = '/dataset/LBP/testing_set/asus_benign

path2 = '/dataset/LBP/net_test/netgear_vulnerable/'


fe = Image_Feature_Extraction.Feature_Extractor('./dataset/')

#files = [file for file in listdir(path) if isfile(join(path, file))]
files = [file for file in listdir(path2) if isfile(join(path2, file))]

for file in files:
    print('file name: ' + file)
    fe.get_hog(path2, file, './dataset/HOG/testing/')
    















