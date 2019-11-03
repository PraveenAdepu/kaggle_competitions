# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:13:14 2017

@author: SriPrav
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import fbeta_score

inDir = 'C:/Users/SriPrav/Documents/R/27Planet'

train_file = inDir + "/input/train_images.csv"
test_file = inDir + "/input/test_images.csv"
test_additional_file = inDir + "/input/test_additional_images.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
test_additional_df = pd.read_csv(test_additional_file)
print(train_df.shape) # (40479, 4)
print(test_df.shape)  # (40669, 2)
print(test_additional_df.shape)  # (20522, 2)

test_all = pd.concat([test_df,test_additional_df])
print(test_all.shape)  # (61191, 2)


train_data_224_3   = np.load(inDir +"/input/train_data_299_3.npy")
train_target_224_3 = np.load(inDir +"/input/train_target_299_3.npy")
train_id_224_3     = np.load(inDir +"/input/train_id_299_3.npy")

test_data_224_3    = np.load(inDir +"/input/test_data_299_3.npy")
test_id_224_3      = np.load(inDir +"/input/test_id_299_3.npy")

train_data_224_3 = train_data_224_3.astype('float32')
#train_data_224_3 = train_data_224_3 / 255
## check mean pixel value
mean_pixel = [103.939, 116.779, 123.68]
for c in range(3):
    train_data_224_3[:, c, :, :] = train_data_224_3[:, c, :, :] - mean_pixel[c]
# train_data /= 255
    
test_data_224_3 = test_data_224_3.astype('float32')
#test_data_224_3 = test_data_224_3 / 255
for c in range(3):
    test_data_224_3[:, c, :, :] = test_data_224_3[:, c, :, :] - mean_pixel[c]
    
labels = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water",
            "haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down",
            "agriculture","road","selective_logging"]

i=10
Modelname = "dense161"
sub_valfile = inDir + '/submissions/Prav.'+Modelname+'.fold' + str(i) + '.csv'
pred_cv = pd.read_csv(sub_valfile)
labels = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
pred_y = np.array(pred_cv[labels])
valindex   = train_df[train_df['CVindices'] == i].index.tolist()  
y_valid = train_target_224_3[valindex,:]
print('F2 Score : ',fbeta_score(y_valid, np.array(pred_y) > 0.2, beta=2, average='samples'))

def optimise_f2_thresholds(y_valid, pred_y, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(pred_y)
    for i in range(17):
      p2[:, i] = (pred_y[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y_valid, p2, beta=2, average='samples')
    return score

  x = [0.20]*17
  for i in range(17):
    best_i2 = 0.00
    best_score = 0.00
    for i2 in range(resolution):
      i2 /= (resolution*1.00)
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x
   
thresholds = optimise_f2_thresholds(y_valid, pred_y) 
validation = pred_cv[labels]
validation1 = validation.apply(lambda x: x > thresholds, axis=1).astype(int)
validation1 = np.array(validation1[labels]) 
print('F2 Score : ',fbeta_score(y_valid, validation1, beta=2, average='samples'))

sub_file = inDir + '/submissions/Prav.'+Modelname+'.fold' + str(i) +'-test'+ '.csv'
pred_test = pd.read_csv(sub_file)

pred_test_threshold = pred_test[labels].apply(lambda x: x > thresholds, axis=1).astype(int)
pred_test_threshold['image_name'] = pred_test['image_name']

sub_file_tags = inDir + '/submissions/Prav.'+Modelname+'.fold' + str(i) + 'flag-test-sub'+'.csv'
pred_test_threshold.to_csv(sub_file_tags, index=False)

tags = []
for r in pred_test_threshold[labels].values:
    r = list(r)
    tags.append(' '.join([j[1] for j in sorted([[r[i],labels[i]] for i in range(len(labels)) if r[i]>0], reverse=True)]))
    
pred_test_threshold['tags'] = tags
pred_test_threshold['image_name'] = pred_test_threshold['image_name'].str.replace('.jpg','')
i=9
sub_file_tags = inDir + '/submissions/Prav.'+Modelname+'_aug.fold' + str(i) + '-test-sub'+'.csv'
pred_test_threshold[['image_name','tags']].to_csv(sub_file_tags, index=False)

##############################################################################################################################

sub_file = inDir + '/submissions/Prav.allmodels.fold0110.csv'
labels = ["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging"]
pred_test = pd.read_csv(sub_file)
pred_test = pred_test[["slash_burn","clear","blooming","primary","cloudy","conventional_mine","water","haze","cultivation","partly_cloudy","artisinal_mine","habitation","bare_ground","blow_down","agriculture","road","selective_logging","image_name"]]

pred_test_threshold = pred_test[labels].apply(lambda x: x > 0, axis=1).astype(int)
pred_test_threshold['image_name'] = pred_test['image_name']

tags = []
for r in pred_test_threshold[labels].values:
    r = list(r)
    tags.append(' '.join([j[1] for j in sorted([[r[i],labels[i]] for i in range(len(labels)) if r[i]>0], reverse=True)]))
    
pred_test_threshold['tags'] = tags
pred_test_threshold['image_name'] = pred_test_threshold['image_name'].str.replace('.jpg','')

sub_file_tags = inDir + '/submissions/Prav.allmodels.fold0110-sub2'+'.csv'
pred_test_threshold[['image_name','tags']].to_csv(sub_file_tags, index=False)

