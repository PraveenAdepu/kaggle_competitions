import numpy as np
import dicom
import glob
from matplotlib import pyplot as plt
import os
import cv2
import mxnet as mx
import pandas as pd
from sklearn import cross_validation


inDir = 'C:/Users/SriPrav/Documents/R/19DSB2017'

resnet50Model = inDir + "/input/preModels/resnet-50"
Stage1SourceFolder = inDir + "/input/sources/stage2/stage2/*"
FeaturesExtraction_numpyFiles = inDir + "/input/sources/stage2/stage2/"
FeatureExtraction_Folder = inDir + "/input/FeatureExtraction_00_stg2"


def get_extractor():
    model = mx.model.FeedForward.load(resnet50Model, 0, ctx=mx.gpu(), numpy_batch_size=1)
    #model = mx.mod.Module.load('C:/Users/SriPrav/Documents/R/19DSB2017/input/model/resnet-50', 0)
    fea_symbol = model.symbol.get_internals()["flatten0_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), 
                                      symbol=fea_symbol ,
                                             numpy_batch_size=64,
                                             arg_params=model.arg_params, 
                                             aux_params=model.aux_params,
                                             allow_extra_params=True
                                             )

    return feature_extractor


def get_3d_data(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def get_data_id(path):
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0
    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))        
    batch = np.array(batch)
    return batch


def calc_features():
    net = get_extractor()
    for folder in glob.glob(Stage1SourceFolder):
        batch = get_data_id(folder)
        feats = net.predict(batch)
        print(feats.shape)
        np.save(folder, feats)

if __name__ == '__main__':
    calc_features()

##########################################################################################################
# Move the feature extraction numpy files to features folder 
##########################################################################################################    
import os
import shutil


files = os.listdir(FeaturesExtraction_numpyFiles)

for f in files:
    if f.endswith('.npy'):
        shutil.move(os.path.join(FeaturesExtraction_numpyFiles,f), os.path.join(FeatureExtraction_Folder,f))   