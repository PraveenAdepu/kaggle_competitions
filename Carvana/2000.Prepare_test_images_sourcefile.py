# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:02:03 2017

@author: SriPrav
"""
import numpy as np
np.random.seed(2017)

import os
import glob
import cv2
import datetime
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

inDir = 'C:/Users/SriPrav/Documents/R/29Carvana'

path = os.path.join('C:\Users\SriPrav\Documents\R\\29Carvana', 'input', 'test', '*.jpg')
files = glob.glob(path)
test_images = pd.DataFrame(files, columns=['img'])
sub_file = inDir + '/input/test_images.csv'
test_images.to_csv(sub_file, index=False)

