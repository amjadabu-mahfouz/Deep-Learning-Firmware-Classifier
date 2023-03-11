# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 04:13:43 2021

@author: user
"""

import mainController as mc


controller = mc.mainController()

#controller.setImagePath('./dataset/REGULAR RGB/testing_set/asus_test/asus_vulnerable/')

#controller.setDestinationPath('./dataset/HOG/testing/')

#controller.setfilter('lbp')

#controller.filter_images()

controller.test_model('1', 'test1_HOG_dlink', './dataset/HOG/dlink_train', './dataset/HOG/dlink_test')

