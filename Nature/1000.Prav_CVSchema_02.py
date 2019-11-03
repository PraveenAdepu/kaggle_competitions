# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 22:13:52 2017

@author: SriPrav
"""

import os
import numpy as np
import shutil
from sklearn.cross_validation import KFold

np.random.seed(2017)

root_train = 'C:/Users/SriPrav/Documents/R/18Nature/input/train_split'
root_val = 'C:/Users/SriPrav/Documents/R/18Nature/input/val_split'

root_total = 'C:/Users/SriPrav/Documents/R/18Nature/input/train'

FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.865
nfolds = 5
random_state = 2017
fish = 'ALB'
num_fold = 0
for fish in FishNames:
    if fish not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, fish))

    total_images = os.listdir(os.path.join(root_total, fish))

    nbr_train = int(len(total_images) * split_proportion)
    kf = KFold(len(total_images), n_folds=nfolds, shuffle=True, random_state=random_state)
    for train_index, test_index in kf:
        X_train = total_images[train_index]
        X_valid = total_images[test_index]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train))
        print('Split valid: ', len(X_valid))
        
    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]

    val_images = total_images[nbr_train:]

    for img in train_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_train, fish, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    if fish not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, fish))

    for img in val_images:
        source = os.path.join(root_total, fish, img)
        target = os.path.join(root_val, fish, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))

 # training samples: 3263, # val samples: 514