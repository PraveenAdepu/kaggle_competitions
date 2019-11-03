# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:16:12 2017

@author: SriPrav
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 03 05:25:33 2017

@author: SriPrav
"""

import bson
import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook

inDir = 'C:/Users/SriPrav/Documents/R/32Cdiscount'

out_folder = inDir + '/input/validation'

# Create output folder
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
	
# Create categories folders
categories = pd.read_csv(inDir +'/input/category_names.csv', index_col='category_id')

for category in tqdm_notebook(categories.index):
    os.mkdir(os.path.join(out_folder, str(category)))
	
              
inDir = 'C:/Users/SriPrav/Documents/R/32Cdiscount'

images_train = pd.read_csv(inDir + '/images_train.csv')

images_train.head()

y, rev_labels = pd.factorize(images_train['image_category'])

images_train['y'] = y

images_train_index = images_train.index.get_values()


build_index, valid_index,y_build, y_valid = train_test_split(images_train_index, y, test_size=0.2, random_state=random_state, stratify=y)

print(len(set(y_build))) # 5270
print(len(set(y_valid))) # 5270

images_build = images_train.ix[list(build_index)] # 9897034
images_valid = images_train.ix[list(valid_index)] # 2474259