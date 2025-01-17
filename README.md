# Firmware Classifier

### Prerequsisites
This application was used for the preprocessing of firmware and images as well as testing numerous convolutional neural network models.  

##### For this application to work properly, the following python setup has to be achieved: 

  - The latest version of Python is installed 

  - Tensorflow 2.3+ and Keras 2.4.0+ are installed in the current environment 

  - Skimage, PIL, cv2 python modules need to be installed in the current environment. 

   

This application uses the same file structure as the one in the GitHub repo; all python classes need to be in the same folder, and the “CNN models” folder needs to be present in current directory. The “Home Router Firmware” folder is there only for future reference and can be removed as it is not used in application.  

The key classes that make up the application are described below: 

### Image_Feature_Extraction.py   

This class is used for extracting the features of image files. The skimage module is the key module; histogram-oriented gradients (HOG), local binary patterns (LBP), Gabor, and the gray level co-occurrence matrix are all image filtering algorithms implemented in Image_Feature_Extraction.py. 

### CNNmodels.py 

This class creates six predetermined convolutional neural network models and stores them in the “CNN models/” folder. These models are numbered and correspond to the models used in the research project.  

###   ImageConverter.py 

This class takes the file name, image type, and image name as parameters then uses them to convert the file to either a grayscale or RGB image. The file_name parameter represents the name and path of the file to be converted to an image, the image_type parameter is a flag that specifies whether the converted image is grayscale (0) or RGB (1), and the image_name is what the converted image will be renamed to.  

### MainController.py  

This class connects all the previous files and used for applying image filters and testing the CNN models on premade image datasets.  

To apply any of the image filters, the first step is to have or create a folder containing the grayscale images of the dataset. The second step is to call the appropriate setter methods in the MainController.py, otherwise the feature extraction calls won't work. Finally, the filter_images() method is called to apply the selected filter; the filter_images method won't work if the parameters are improperly set in the second step. 

In order to perform testing on the models, the test_model() method needs to be called with the appropriate parameters. The first parameter is the model number and is used to reference the premade model in the “CNN models/” folder. The second parameter is the model's name and is used to save the model under a different name after training. The third and fourth parameters are for the paths of the training and testing datasets respectively.  


## Using the Main Controller 

##### The first step is to import and initialize the main controller. 

> import mainController as mc 

> controller = mc.mainController() 

### Filtering images 

##### Before the images can be filtered, the appropriate setter methods need to be called. These steps are described below. 

##### Firstly, the folder containing the grayscale dataset has to be set. 

> controller.setImagePath('./dataset/REGULAR RGB/testing_set/asus_test/asus_vulnerable/') 

 

##### Next, the destination folder in which the filtered images are written to needs to be set. 

> controller.setDestinationPath('./dataset/HOG/testing/') 

 

##### The type of filtering algorithm used needs to be set with the following command.  

> controller.setfilter('lbp') 

 

##### The last step is to call the filter_images() method which will use the previously set parameters to apply the appropriate filter.  

> controller.filter_images() 

 

 

### Testing CNN models 

##### Testing the models is simple and requires only one method call to test_model(). The testing and training datasets have to be made beforehand and their paths supplied as parameters. Refer to the section above on the MainControlly.py class for more details on the parameters.  

> controller.test_model('1', 'test1_Gabor', './dataset/HOG/asus_train', './dataset/HOG/asus_test') 

 
