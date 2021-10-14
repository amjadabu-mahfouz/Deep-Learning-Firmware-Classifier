# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 02:46:42 2021

@author: user
"""

from os import listdir
from os.path import isfile, join

import ImageConverter
import Image_Feature_Extraction
import CNNmodels



class mainController :
    def __init__(self):
        self.image_path = ''
        self.destination_folder = ''
        self.fe = Image_Feature_Extraction.Feature_Extractor('./')
        self.filter_type = filter_type = ''
        self.models = CNNmodels.CNNmodels()
        self.models.make_Models()
        
    
    def setImagePath(self, image_path):
        self.image_path = image_path
        
    def setDestinationPath(self, destination_path):
        self.destination_path = destination_path
        
    def setfilter(self, filter_type):
        self.filter_type = filter_type
        
        
    def setVulnerabilityFlag(self, is_vulnerable):
        self.is_vulnerable = is_vulnerable
            
    def setGlcmDestination(self, csv_path):
        self.csv_path = csv_path
   
    def filter_images(self):
        files = [file for file in listdir(self.image_path) if isfile(join(self.image_path, file))]
        for file in files:
            print('file name: ' + file)
            self.filter_selection(file)
        
    def filter_selection(self, image_name):
        if self.filter_type == 'glcm':
            self.fe.get_glcm(self.image_path, image_name, self.destination_path_img, self.csv_path, self.is_vulnerable)
        elif self.filter_type == 'hog':        
            self.fe.get_hog(self.image_path, image_name, self.destination_path)
        elif self.filter_type == 'lbp':
            self.fe.get_lbp(self.image_path, image_name, self.destination_path)
        elif self.filter_type == 'gabor':
            self.fe.get_gabor(self.image_path, image_name, self.destination_path)


    
    def test_model(self, model_number, model_name, training_set_path, testing_set_path): 
        from keras.models import load_model
        from keras.preprocessing.image import ImageDataGenerator  
        from PIL import Image, ImageFile
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        model = load_model('./CNN models/Model-' + model_number + '.h5')
        
        # this is the model used to pre-process our images into a format to be used with the Conv.net
        train_datagen = ImageDataGenerator(  rescale=1./255,
                                             #shear_range=0.2,
                                             zoom_range=0.002,
                                             #horizontal_flip=True,
                                             )


        training_set = train_datagen.flow_from_directory(training_set_path,  
                                                 target_size = (500, 500),  
                                                 batch_size = 5,  
                                                 class_mode = 'binary',
                                                 color_mode = 'grayscale')  

        test_datagen = ImageDataGenerator(rescale = 1./255,
                                          zoom_range=0.002,) 

        test_set = test_datagen.flow_from_directory(testing_set_path,  
                                            target_size = (500, 500),  
                                            batch_size = 5,  
                                            class_mode = 'binary', 
                                            color_mode = 'grayscale')   


        history = model.fit(training_set, validation_data = test_set, epochs = 20) 
     
        
        from matplotlib import pyplot as plt
        #history = model1.fit(train_x, train_y,validation_split = 0.1, epochs=50, batch_size=4)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        model.save(('./CNN models/' + model_name + '.h5'))
     
        