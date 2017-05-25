#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:20:08 2017

@author: jparker
"""
# Prior to running this script make sure you arein the cv environment
# This is needed for the cv2 package. To do this:
# 1. In LXTerminal Type "source activate cv" Enter
# 2. Type "spyder" Enter. Spyder will launch and you can run this code

import os
import cv2

dir = 'train' # Specifiy your directory here (e.g., train or test)

#os.system("cp -rf "+dir+" "+dir+"canny") # Uncomment to make a new directory
for directory, subdirectories, files in os.walk(dir):
    for file in files:
        img_rgb = cv2.imread(os.path.join(directory,file))
        img_canny = cv2.Canny(img_rgb,100,200, apertureSize = 3,L2gradient = True)
        img_canny = (255-img_canny)
        filename = os.path.splitext(file)[0]
        cv2.imwrite(os.path.join(directory,(filename +".png")),img_canny)
#        os.remove(os.path.join(directory,file)) # Removes the original files (for train)
        
print(".....DONE WITH THE CANNY IMAGES.......")
#
#
#num_down = 2       # number of downsampling steps
#num_bilateral = 7  # number of bilateral filtering steps
#
#os.system("cp -rf "+dir+" "+dir+"edge")
#for directory, subdirectories, files in os.walk(dir+"edge"):
#    for file in files:
#        img_rgb = cv2.imread(os.path.join(directory,file))
#        
#        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
#        img_blur = cv2.medianBlur(img_gray, 3)
#        img_edge = cv2.adaptiveThreshold(img_blur, 255,
#                                 cv2.ADAPTIVE_THRESH_MEAN_C,
#                                 cv2.THRESH_BINARY, 9, 6)      
#        
#        cv2.imwrite(os.path.join(directory,(filename +".png")),img_edge)
#        os.remove(os.path.join(directory,file)) 
#        
#print(".....DONE WITH THE EDGE DETECTOR IMAGES......")
print(".....ALL DONE................................")