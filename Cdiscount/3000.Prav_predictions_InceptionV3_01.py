# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 07:29:39 2017

@author: SriPrav
"""
#import os
#os.environ["KERAS_BACKEND"] = "tensorflow"

import os, sys, math, io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct

%matplotlib inline
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from collections import defaultdict
from tqdm import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

keras.__version__, tf.__version__

data_dir = "C:/Users/SriPrav/Documents/R/32Cdiscount/input/"

train_bson_path = os.path.join(data_dir, "train.bson")
num_train_products = 7069896

# train_bson_path = os.path.join(data_dir, "train_example.bson")
# num_train_products = 82

test_bson_path = os.path.join(data_dir, "test.bson")
num_test_products = 1768182

categories_path = os.path.join(data_dir, "category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

#categories_df.to_csv("categories.csv")
categories_df.head()

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

cat2idx, idx2cat = make_category_tables()

# Test if it works:
cat2idx[1000012755], idx2cat[0]

def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df

#%time train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)
#
#
#train_offsets_df.head()
#
##train_offsets_df.to_csv("train_offsets.csv")
#
## How many products?
#len(train_offsets_df)
#
## How many categories?
#len(train_offsets_df["category_id"].unique())
#
## How many images in total?
#train_offsets_df["num_imgs"].sum()

#def make_val_set(df, split_percentage=0.2, drop_percentage=0.):
#    # Find the product_ids for each category.
#    category_dict = defaultdict(list)
#    for ir in tqdm(df.itertuples()):
#        category_dict[ir[4]].append(ir[0])
#
#    train_list = []
#    val_list = []
#    with tqdm(total=len(df)) as pbar:
#        for category_id, product_ids in category_dict.items():
#            category_idx = cat2idx[category_id]
#
#            # Randomly remove products to make the dataset smaller.
#            keep_size = int(len(product_ids) * (1. - drop_percentage))
#            if keep_size < len(product_ids):
#                product_ids = np.random.choice(product_ids, keep_size, replace=False)
#
#            # Randomly choose the products that become part of the validation set.
#            val_size = int(len(product_ids) * split_percentage)
#            if val_size > 0:
#                val_ids = np.random.choice(product_ids, val_size, replace=False)
#            else:
#                val_ids = []
#
#            # Create a new row for each image.
#            for product_id in product_ids:
#                row = [product_id, category_idx]
#                for img_idx in range(df.loc[product_id, "num_imgs"]):
#                    if product_id in val_ids:
#                        val_list.append(row + [img_idx])
#                    else:
#                        train_list.append(row + [img_idx])
#                pbar.update()
#                
#    columns = ["product_id", "category_idx", "img_idx"]
#    train_df = pd.DataFrame(train_list, columns=columns)
#    val_df = pd.DataFrame(val_list, columns=columns)   
#    return train_df, val_df
#
#train_images_df, val_images_df = make_val_set(train_offsets_df, split_percentage=0.2, 
#                                              drop_percentage=0.)
#
#train_images_df.head()
#
#val_images_df.head()
#
#print("Number of training images:", len(train_images_df))
#print("Number of validation images:", len(val_images_df))
#print("Total images:", len(train_images_df) + len(val_images_df))
#
#len(train_images_df["category_idx"].unique()), len(val_images_df["category_idx"].unique())
#
#category_idx = 619
#num_train = np.sum(train_images_df["category_idx"] == category_idx)
#num_val = np.sum(val_images_df["category_idx"] == category_idx)
#num_val / num_train

#train_images_df.to_csv("train_images.csv")
#val_images_df.to_csv("val_images.csv")

#categories_df = pd.read_csv("categories.csv", index_col=0)
#cat2idx, idx2cat = make_category_tables()
#
#train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)
#train_images_df = pd.read_csv("train_images.csv", index_col=0)
#val_images_df = pd.read_csv("val_images.csv", index_col=0)

#train_offsets_df.head()
#
#train_images_df.head()
#
#train_images_df = train_images_df.sample(frac=1).reset_index(drop=True)
#
#train_images_df.head()
#
#val_images_df.head()

#Uses LabelEncoder for class_id encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(pd.read_csv(categories_path).category_id)

#Testing the encoder
original=le.classes_[:5]
print("5 original classes:", original)
encoded=le.transform(original)
print("5 encoded classes:",encoded)
print("getting back the original classes:", le.inverse_transform(encoded))

#def create_bin_file(images_df, offsets_df, bson_file_name, bin_file_name, encoder):
#    with open(bson_file_name, 'rb') as bson_file, open(bin_file_name, 'wb') as bin_file:    
#        #uses Human Analog previously created dataframes
#        for index, row in images_df.iterrows():
#            offset_row = offsets_df.loc[row.product_id]
#            bson_file.seek(offset_row["offset"])
#            item_data = bson_file.read(offset_row["length"])
#
#            # Grab the image from the product.
#            item = bson.BSON.decode(item_data)
#            img_idx = row["img_idx"]
#            bson_img = item["imgs"][img_idx]["picture"]
#
#            #write down the encoded class, the size of the img and the img it self 
#            encoded_class = encoder.transform([offset_row.category_id])[0]
#            img_size = len(bson_img)
#            bin_file.write(struct.pack('<ii', encoded_class, img_size))   
#            bin_file.write(bytes(bson_img))   
#        bin_file.close()
#        bson_file.close()
#        
##test function
#def bin_file_test(file_name, encoder, n=3):
#    with open(file_name, 'rb') as bin_file:    
#        count = 0
#        while count<n:
#            count += 1 
#            buffer=bin_file.read(8)
#            encoded_class, length = struct.unpack("<ii", buffer)
#            bson_img = bin_file.read(length)
#            img = load_img(io.BytesIO(bson_img), target_size=(180,180))
#            plt.figure()
#            plt.imshow(img)
#            plt.text(5,20,
#                     "%d Class: %s (size: %d)" %(count, encoder.inverse_transform(encoded_class), length),
#                    backgroundcolor='0.75',alpha=.5)
            

##create train bin file and test it!!!
#img_df = train_images_df#[:1000] #remove this in production environment
#create_bin_file(img_df, train_offsets_df, train_bson_path, 'train.bin', le)
#bin_file_test('train.bin', le, n=9)
#
##create val bin file and test it
#img_df = val_images_df#[:1000] #remove this in production environment
#create_bin_file(img_df, train_offsets_df, train_bson_path, 'val.bin', le)
#bin_file_test('val.bin', le)


from keras.preprocessing import image
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import threading

#The generator. The flow method does the generator job!
class BinFileIterator(Iterator):
    def __init__(self, bin_file_name, img_generator, samples, 
                 target_size=(180,180), 
                 batch_size=32, num_class=5270):
        self.file = open(bin_file_name,'rb')
        self.img_gen=img_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)
        self.num_class = num_class
        self.lock = threading.Lock() #Since we have 2 files, each generator has its own lock
        super(BinFileIterator, self).__init__(samples, batch_size, shuffle=False, seed=None)

    def flow(self, index_array):
        X = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        Y = np.zeros((len(index_array), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            with self.lock:
                buffer=self.file.read(8)
                if len(buffer) < 8:
                    self.file.seek(0)
                    buffer=self.file.read(8)
                encoded_class, length = struct.unpack("<ii", buffer)
                bson_img = self.file.read(length)
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)
            x = image.img_to_array(img)
            #x = self.img_gen.random_transform(x)
            #x = self.img_gen.standardize(x)
            X[i] = x
            Y[i, encoded_class] = 1
        X = preprocess_input(np.array(X))
        return X, Y

    def next(self):
        with self.lock: 
            index_array = next(self.index_generator)
        return self.flow(index_array[0])
    
#train_img_gen = ImageDataGenerator(rescale=1./255)
#data_gen = BinFileIterator('train.bin', img_generator=train_img_gen,  samples=1000,
#                 target_size=(180,180), 
#                 batch_size=10)
#
#for b in range(10):
#  imgs, categories = data_gen.next()
#  for img, category in zip(imgs, categories): 
#      plt.figure()
#      plt.imshow(img)
#      plt.text(5,20,
#               "Class: %d %s" % (np.argmax(category), le.inverse_transform(np.argmax(category))),
#               backgroundcolor='0.75',alpha=.5)
      
import time
#data_gen = BinFileIterator('train.bin', img_generator=train_img_gen,  samples=1000,
#                 target_size=(180,180), 
#                 batch_size=128) #We changed the batch size here 
#for b in range(3):
#  %time imgs, categories = data_gen.next()
#  print("Retrieved: %d" %(len(imgs))  ) 
#
#
#plt.figure()
#plt.imshow(imgs[-1])
#plt.text(5,20,
#        "Class: %d %s" % (np.argmax(categories[-1]), le.inverse_transform(np.argmax(categories[-1]))),
#        backgroundcolor='0.75',alpha=.5)

import _thread


##Lets use a large batch size
#data_gen = BinFileIterator('train.bin', img_generator=train_img_gen,  samples=1000,
#                 target_size=(180,180), 
#                 batch_size=450) #We changed the batch size here 
#
## Define a function for the thread
#def execute_batch(t_name):
#   imgs, categories = data_gen.next()
#   print(t_name, "retrieved: %d" %len(imgs), 
#                 "last category:" , le.inverse_transform(np.argmax(categories[-1])))
#
## Create two threads as follows
#try:
#   _thread.start_new_thread( execute_batch, ("Thread-1", ) )
#   time.sleep(0.001)   
#   _thread.start_new_thread( execute_batch, ("Thread-2", ) )
#   time.sleep(0.001)   
#   _thread.start_new_thread( execute_batch, ("Thread-3", ) )
#except:
#   print ("Error: unable to start thread")
#
#time.sleep(5)

#################################################################################################################################
random_state = 2017
np.random.seed(random_state)               
inDir = 'C:/Users/SriPrav/Documents/R/32Cdiscount'
MODEL_WEIGHTS_FILE = inDir + '/Prav_Inceptionv3_01.h5'


num_classes = 5270  # This will reduce the max accuracy to about 0.75
CHANNELS = 3
ROWS = 180
COLUMNS = 180
nb_epoch = 1
VERBOSEFLAG = 1
batch_size  = 64
patience = 2
optim_type = 'Adam'
learning_rate = 1e-3

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.models import Model
from ipywidgets import IntProgress
from keras.metrics import top_k_categorical_accuracy
import time
import glob
import math
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,Callback, ModelCheckpoint
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

################################################
print("create the base pre-trained model")
################################################
input = Input(shape=(ROWS, COLUMNS,CHANNELS), name='NEW_image_input_180x180X3') #New input layer, good to the competition shape
base_model = InceptionV3(input_tensor=input, weights='imagenet', include_top=False)

for layer in base_model.layers[1:]:
    layer.trainable = False

x = base_model.output
#Some aditional layers
x = GlobalAveragePooling2D(name = 'NEW_GlobalAveragePooling2D')(x)
# let's add a fully-connected layer
#x = Dense(1024, activation='relu', name='NEW_Dense_1024')(x)
#x = Dense(2048, activation='relu', name='NEW_Dense_2048')(x)
# and a logistic layer -- let's say we have 200 classes

#modelo novo
predictions = Dense(num_classes, activation='softmax', name='NEW_Predictions_5270')(x)
model = Model(inputs=base_model.input, outputs=predictions)

################################################
print("fit the new classifier")
################################################

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers

#for layer in base_model.layers[1:]:
#    layer.trainable = False
    


callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True, verbose=VERBOSEFLAG),
                ]
#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
if optim_type == 'SGD':
    optim = SGD(lr=learning_rate, momentum=0.9)
else:
    optim = Adam(lr=learning_rate)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#train_img_gen = ImageDataGenerator() #Configure as you want
##train_images_df = 9902648
##val_images_df = 2468645
#train_gen = BinFileIterator('train.bin', img_generator=train_img_gen,  samples=2500000,
#                 target_size=(180,180), 
#                 batch_size=batch_size)  
#
#val_img_gen = ImageDataGenerator() #Configure as you want
#val_gen = BinFileIterator('val.bin', img_generator=val_img_gen,  samples=500000,
#                 target_size=(180,180), 
#                 batch_size=batch_size) 
#   
#model.fit_generator( generator=train_gen,                            
#                     steps_per_epoch = math.ceil(2500000 / batch_size), 
#                     nb_epoch = nb_epoch, 
#                     callbacks = callbacks,
#                     validation_data=val_gen, 
#                     validation_steps = math.ceil(500000 / batch_size), 
#                     max_q_size=10,
#                     workers = 4,
#                     verbose = VERBOSEFLAG 
#                  )

################################################
print("fine tune the model")
################################################


for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True
   
nb_epoch = 2
initial_epoch = 1
learning_rate = 1e-3
optim = Adam(lr=learning_rate)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit_generator( generator=train_gen,                            
#                     steps_per_epoch = math.ceil(9902648 / batch_size), 
#                     initial_epoch = initial_epoch,
#                     nb_epoch = nb_epoch, 
#                     callbacks = callbacks,
#                     validation_data=val_gen, 
#                     validation_steps = math.ceil(2468645 / batch_size), 
#                     max_q_size=10,
#                     workers = 4,
#                     verbose = VERBOSEFLAG 
#                  )

model.load_weights(MODEL_WEIGHTS_FILE)
################################################################################################################

import io
import bson
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import cpu_count

images_test = pd.read_csv(inDir + '/images_test.csv')
images_test.dtypes

images_test = images_test.sort_values(['_id', 'image_id'], ascending=[True, True])
images_test = images_test[images_test.image_id == 0]
images_test[:10]

def get_im_cv2(path):
    # path = 'C:\\Users\\SriPrav\\Documents\\R\\32Cdiscount\\input\\test_images\\10-0.jpg'
    img = cv2.imread(path)
    #plt.imshow(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
    
def normalize_image(x):
    x = np.array(x, dtype=np.uint8)
    x = x.astype('float32')
    x = preprocess_input(x)
    return x
    
#def load_train_frombatch(images_batch):
#   
#    X_test = []
##    start_time = time.time()
#    for fl in images_batch.image_path.values:
##        print(fl)        
#        img = get_im_cv2(fl)        
#        X_test.append(img)    
##    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
#    return X_test
#
#    
#def batch_generator_test(images_build ,batch_size, shuffle):
#    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
#    number_of_batches = np.ceil(len(images_build)/batch_size)
#    counter = 0
#    sample_index = images_build.index.get_values()
##    if shuffle:
##        np.random.shuffle(sample_index)
#    while True:
#        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
#        images_build_batch = images_build.ix[list(batch_index)]
#        X_batch, y_batch = load_train_frombatch(images_build_batch)
#        X_batch = normalize_image(X_batch)
#        counter += 1
#        yield X_batch, y_batch
#        if (counter == number_of_batches):
#            if shuffle:
#                np.random.shuffle(sample_index)
#            counter = 0

def get_test_images(images_file):
    X_test = []
    count = 0
    for image in images_test['image_path']:        
        img = get_im_cv2(image)
        X_test.append(img)
        count +=1
        if count % 100000 == 0:
            print(count)
    return X_test

X_test = get_test_images(images_test)


def predictions_test(images_file):
    prediction_category = []
    count=0  
    for image_idx in range(177):
        #print(10000*image_idx)
        # path = 'C:\\Users\\SriPrav\\Documents\\R\\32Cdiscount\\input\\test_images\\10-0.jpg'
        img = X_test[10000*image_idx:10000*(image_idx+1)]
        img = normalize_image(img)
        pred_category = model.predict(img)
        preds_category = pred_category.argmax(1)
        preds_category_labels = le.inverse_transform(preds_category)
        #pred_category = le.inverse_transform(np.argmax(model.predict(img)[0]))
        prediction_category.append(preds_category_labels)
        count +=1
        if count % 10 == 0:
            print(count)
    return prediction_category
#del prediction_categories, pred_category, preds_category, preds_category_labels
#del prediction_categories
prediction_categories = predictions_test(images_test)


prediction_category_list = pd.DataFrame()
for list_idx in range(177):
    current_list = pd.DataFrame()
    current_list = pd.DataFrame(prediction_categories[list_idx].transpose())
    prediction_category_list = prediction_category_list.append(current_list)


prediction_category_list = prediction_category_list.reset_index(drop=True)

prediction_category_list.head()


images_test_original = images_test    
images_test['category_id'] = prediction_category_list.values
images_test = images_test.reset_index(drop=True)

images_test.head()
images_test[['_id','category_id']].to_csv("Prav_Inceptionv3_01.csv", index=False)
#
#def predictions(images_file):
#    prediction_category = []
#    count=0  
#    for image in images_test['image_path']:
#        # path = 'C:\\Users\\SriPrav\\Documents\\R\\32Cdiscount\\input\\test_images\\10-0.jpg'
#        img = get_im_cv2(image)
#        img = normalize_image(img)
#        pred_category = le.inverse_transform(np.argmax(model.predict(img[None])[0]))
#        prediction_category.append(pred_category)
#        count +=1
#        if count % 100000 == 0:
#            print(count)
#    return prediction_category
#
#
#prediction_categories = predictions(images_test)
#prediction_categories_df = pd.DataFrame(prediction_categories, columns = ["category_id"])
#
#images_test['category_id'] = prediction_categories_df['category_id']
#
#images_test.to_csv("Prav_Inceptionv3_01_allImages.csv", index=False)
#    
    





















import io
import bson
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import concurrent.futures
from multiprocessing import cpu_count

num_cpus = 24

def imread(buf):
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_ANYCOLOR)

def img2feat(im):
    x = cv2.resize(im, (ROWS, COLUMNS), interpolation=cv2.INTER_AREA)
    x = np.float32(x)
    x = preprocess_input(x)
    return x
    #return np.float32(x) / 255

def load_image(pic, target, bar):
    picture = imread(pic)
    x = img2feat(picture)
    bar.update()
    
    return x, target

    
submission = pd.read_csv(inDir + '/input/sample_submission.csv', index_col='_id')

most_frequent_guess = 1000018296
submission['category_id'] = most_frequent_guess # Most frequent guess


num_images_test = 1768182  # We only have time for a few test images..

bar = tqdm_notebook(total=num_images_test * 2)
with open(inDir+'/input/test.bson', 'rb') as f, \
         concurrent.futures.ThreadPoolExecutor(num_cpus) as executor:

    data = bson.decode_file_iter(f)

    future_load = []
    
    for i,d in enumerate(data):
        if i >= num_images_test:
            break
        future_load.append(executor.submit(load_image, d['imgs'][0]['picture'], d['_id'], bar))

    print("Starting future processing")
    for future in concurrent.futures.as_completed(future_load):
        x, _id = future.result()
        
        y_cat = le.inverse_transform(np.argmax(model.predict(x[None])[0]))
        if y_cat == -1:
            y_cat = most_frequent_guess

        bar.update()
        submission.loc[_id, 'category_id'] = y_cat
print('Finished')

submission.to_csv(inDir+'/submissions/Prav_InceptionV3_01.csv')


future_load[0]

#####################################################################################################################

from keras import backend as K
#from keras.preprocessing.image import ImageDataGenerator

submission_df = pd.read_csv(data_dir + "sample_submission.csv")
submission_df.head()

#test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
data = bson.decode_file_iter(open(test_bson_path, "rb"))
num_images_test = 1768182
with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])

        batch_x = np.zeros((num_imgs, ROWS, COLUMNS, CHANNELS), dtype=K.floatx())

        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]

            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(ROWS, COLUMNS))
            x = img_to_array(img)
            #x = test_datagen.random_transform(x)
            #x = test_datagen.standardize(x)

            # Add the image to the batch.
            batch_x[i] = x
        batch_x = preprocess_input(np.array(batch_x))
        prediction = model.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)

        submission_df.iloc[c]["category_id"] = le.inverse_transform(cat_idx)#idx2cat[cat_idx]        
        pbar.update()

submission_df.to_csv(inDir+'/submissions/Prav_InceptionV3_01.csv', index=False)

