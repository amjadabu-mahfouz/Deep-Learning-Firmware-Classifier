# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:05:54 2021

@author: user
"""

from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops

from skimage.feature import local_binary_pattern

from skimage.measure import moments_central
from skimage.measure import moments_normalized
from skimage.measure import moments_hu

from PIL import Image, ImageOps
from PIL.ImageStat import Stat

import cv2
import numpy as np

import zipfile, os
import csv

class Feature_Extractor : 
    def __init__ (self, destination_folder):
        # starting parameters
        #destination_folder = ''
        #image_location = './dataset/test_set/benign/'
        #image_name =  'DSL-2640B.exe.png'
        #is_vulnerable = 0
        self.destination_folder = destination_folder
      
        
        
        
        
        
        
        """
        HOG image filtering 
        """
    def get_hog(self, image_path, image_name, destination_path):  
        # extract image into grayscale format
        img = cv2.imread(image_path + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
       
        
        # extract HOG features
        feature_descriptor, hog_img = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        	cells_per_block=(3, 3), visualize=True, multichannel=False)
        
        # save the image
        cv2.imwrite(destination_path + image_name + '.png', hog_img)
        
        
       
        
        """
        Gray Co-Occurance Matrix
        """
    def get_glcm (self, image_path, image_name, destination_path_img, destination_path_csv, is_vulnerable): 
        img = cv2.imread(image_path + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
     
        
        result = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], normed=True)
        
        # get all matricies for the same distance (simplify the 4D array and remove the distance element)
        x = result[:, :, 0, :]
        
        # take the averages of all elements in all the directional matricies, so we can then normalize them
        avg_matrix = np.mean( [x[:,:,0], x[:,:,1], x[:,:,2], x[:,:,3]], axis=0)
        
        #normalize the values of the matricies so they are rescaled to 0-255 (so then we can turn them into grayscale images)
        norm_matrix = avg_matrix * 255.0/avg_matrix.max()
        norm_matrix = norm_matrix.astype(int)
        
        # turn the 255/255 normalized matrix into a grayscale image
        norm_matrix = Image.fromarray(norm_matrix)
        
        # save the image
        cv2.imwrite(destination_path_img + image_name + '.png', norm_matrix)
        
        
        
        # save the extracted features into a csv format
        contrast = greycoprops(result, prop = 'contrast')
        
        dissimilarity = greycoprops(result, prop = 'dissimilarity')
        
        homogeneity = greycoprops(result, prop = 'homogeneity')
        
        energy = greycoprops(result, prop = 'energy')
        
        correlation = greycoprops(result, prop = 'correlation')
        
        ASM = greycoprops(result, prop = 'ASM')
        
        
        row_list = [image_name, contrast[0,0], contrast[0,1], contrast[0,2], contrast[0,3], 
                    dissimilarity[0,0], dissimilarity[0,1], dissimilarity[0,2], dissimilarity[0,3], 
                    homogeneity[0,0], homogeneity[0,1], homogeneity[0,2], homogeneity[0,3], 
                    energy[0,0], energy[0,1], energy[0,2], energy[0,3], 
                    correlation[0,0], correlation[0,1], correlation[0,2], correlation[0,3], 
                    ASM[0,0], ASM[0,1], ASM[0,2], ASM[0,3],
                    is_vulnerable]
        
        # save the csv
        with open(destination_path_csv + 'glcm_dataset.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_list)
        
        
        
        
        
        """
        LBP Matrix -(use all methods)
        """
    def get_lbp (self, image_path, image_name, destination_path): 
        img = cv2.imread(image_path + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
       
        # method = default
        lbp = local_binary_pattern(img, 8, 1, method='default')
        
        lbp = lbp.astype(np.uint8)
        lbp = Image.fromarray(lbp)
        # save the LBP filtered image
        lbp.save(destination_path + image_name + '.png')
        
        """
        # method = var
        lbp = local_binary_pattern(img, 8, 1, method='var')
        
        lbp = lbp.astype(np.uint8)
        lbp = Image.fromarray(lbp)
        # save the LBP filtered image
        lbp.save(self.destination_folder + '/LBP/var/' + test_or_train  + path_ext + image_name + '.png')
        
        
        # method = ror
        lbp = local_binary_pattern(img, 8, 1, method='ror')
        
        lbp = lbp.astype(np.uint8)
        lbp = Image.fromarray(lbp)
        # save the LBP filtered image
        lbp.save(self.destination_folder + '/LBP/ror/' + test_or_train  + path_ext + image_name + '.png')
        
        
        
        # method = uniform
        lbp = local_binary_pattern(img, 8, 1, method='uniform')
        
        lbp = lbp.astype(np.uint8)
        lbp = Image.fromarray(lbp)
        # save the LBP filtered image
        lbp.save(self.destination_folder + 'LBP/uniform/' + test_or_train  + path_ext + image_name + '.png')
        """
        
        
        
        """
        Gabor Filter
        """
    def get_gabor (self, image_path, image_name, destination_path): 
        img = cv2.imread(image_path + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
      
        
        # this is the kernel/filter size (x by x pixel grid/matrix to be used as a filter)
        ksize = 3
        
        # use small sigma on small features (using large sigma on small features will completely miss them)
        sigma = 3
        
        # 1/4 for horizontal alignment ... 3/4 for other horizontal (so if your filter is horizontal, it will block all vertical stuff/features)  (default = 1*np.pi/2)
        theta = 1*np.pi/2
        
        # 1/4 works best for angled features
        lamda = 1*np.pi
        
        # this is for spect ratio of kernel ... making it 0 will make it a very thin/narrow ... and 1 will make it a cricle
        gamma = 0.5
        
        # this is for the phase shit or the cemmetrical offset
        phi = 0
        
        # this is to declare gobor filter/kernel
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi)
        
        # this is to apply the gabor filter to the image
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        
        # save the filtered image
        cv2.imwrite(destination_path + image_name + '.png', filtered_img)

        
        
        """
        Hu Moments - invariant to size, rotation, etc..
        """
    def get_hu_moments (self, image_path, image_name, destination_path, is_vulnerable): 
        img = cv2.imread(image_path + image_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        central_moments = moments_central(img)
        moments_normed = moments_normalized(central_moments)
        
        h_m = moments_hu(moments_normed)
        
        row_list = [image_name, h_m[0], h_m[1], h_m[2], h_m[3], h_m[4], h_m[5], h_m[6], is_vulnerable]
        
        # save the csv
        with open(self.destination_folder + 'HU_MOMENTS/hu_moments.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_list)
        
        
        
        
        """
        Extracting image statistics
        """
    def get_image_stats (self, image_path, image_name, destination_path, is_vulnerable): 

               
        img_stt = Image.open(image_path + image_name)
        
        sts = Stat(img_stt)
        
        
        # these stats are taken from bands in image
        pixel_count = sts._getcount()
        pixel_count = pixel_count[0]
        
        band_sum = sts._getsum()
        band_sum = band_sum[0]
        
        band_sum_squared = sts._getsum2()
        band_sum_squared = band_sum_squared[0]
        
        mean = sts._getmean()
        mean = mean[0]
        
        median = sts._getmedian()
        median = median[0]
        
        root_mean_square = sts._getrms()
        root_mean_square = root_mean_square[0]
        
        standard_dev = sts._getstddev()
        standard_dev = standard_dev[0]
        
        variance = sts._getvar()
        variance = variance[0] 
        
        
        row_list = [image_name, pixel_count, band_sum, band_sum_squared, mean, median,
                    root_mean_square, standard_dev, variance, is_vulnerable]
        
        
        with open(destination_path + 'image_stats.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(row_list)
        
