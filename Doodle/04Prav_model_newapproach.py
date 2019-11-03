# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:51:02 2018

@author: SriPrav
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import os
# import multiprocessing
import cv2
import math


from keras.losses import sparse_categorical_crossentropy
from tqdm import trange

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input


plt.rcParams["figure.max_open_warning"] = 300

inDir = r"C:/Users/SriPrav/Documents/R/55Doodle"
INPUT_DIR = inDir+'/input'
sample_size = 30000

def strokes_to_img(in_strokes):
    in_strokes = eval(in_strokes)
    # make an agg figure
    fig, ax = plt.subplots()
    for x,y in in_strokes:
        ax.plot(x, y, linewidth=12.) #  marker='.',
    ax.axis('off')
    fig.canvas.draw()
    
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    return (cv2.resize(X, (96, 96)) / 255.)[::-1]

class_files = os.listdir(inDir+"/input/train_simplified/")
classes = {x[:-4]:i for i, x in enumerate(class_files)}
to_class = {i:x[:-4].replace(" ", "_") for i, x in enumerate(class_files)}

dfs = [pd.read_csv(inDir+"/input/train_simplified/" + x, nrows=sample_size)[["word", "drawing"]] for x in class_files]
df = pd.concat(dfs)
del dfs


from sklearn.cross_validation import train_test_split

train, valid = train_test_split(df, test_size = 0.01, stratify=df["word"],random_state=1987)

valid.groupby(['word']).agg(['count'])
del df
df = train.copy()
del train

# mppool = multiprocessing.Pool(6)
n_samples = df.shape[0]
size = 96
channels = 3
NCATS = len(classes)
MODEL_WEIGHTS_FILE = inDir + '/Prav_Model04.h5'
batch_size  = 256
patience = 3
optim_type = 'Adam'
learning_rate = 2e-3
VERBOSEFLAG = 1
nb_epoch = 5

pick_order = np.arange(n_samples)
pick_per_epoch = math.ceil(n_samples // batch_size)

def train_gen():
    while True:  # Infinity loop
        np.random.shuffle(pick_order)
        for i in range(pick_per_epoch):
            c_pick = pick_order[i*batch_size: (i+1)*batch_size]
            dfs = df.iloc[c_pick]
            out_imgs = list(map(strokes_to_img, dfs["drawing"]))
            X = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
            y1 = np.array([classes[x] for x in dfs["word"]])
            y = keras.utils.to_categorical(y1, num_classes=NCATS)
            yield X, y
            
train_datagen = train_gen()
x,y = next(train_datagen)

# Display some images
for i in range(12):
    plt.subplot(2,6,i+1)
    plt.imshow(x[i])
    plt.axis('off')
plt.show()

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

def list_all_categories():
    files = os.listdir(os.path.join(INPUT_DIR, 'train_simplified'))
    return sorted([f2cat(f) for f in files], key=str.lower)

def apk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def mapk(actual, predicted, k=3):
    """
    Source: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def preds2catids(predictions):
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

valid_imgs = list(map(strokes_to_img, valid["drawing"]))
x_valid = np.array(valid_imgs)[:, :, :, :3].astype(np.float32)
y_valid = np.array([classes[x] for x in valid["word"]])

y_valid = keras.utils.to_categorical(y_valid, num_classes=NCATS)
  
import gc
gc.collect()

model = MobileNet(input_shape=(size, size, channels), alpha=1., weights=None, classes=NCATS)

callbacks = [
#        EarlyStopping(monitor='val_categorical_accuracy', patience=patience, verbose=VERBOSEFLAG),
        ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=patience,min_delta=0.005, mode='max', cooldown=3, verbose=VERBOSEFLAG),
        ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_categorical_accuracy', save_best_only=True, verbose=VERBOSEFLAG),
                ]


#sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
if optim_type == 'SGD':
    optim = SGD(lr=learning_rate, momentum=0.9)
else:
    optim = Adam(lr=learning_rate)
    
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())

   
model.fit_generator( generator=train_datagen,                            
                     steps_per_epoch = math.ceil(n_samples // batch_size), 
                     nb_epoch = nb_epoch, 
                     callbacks = callbacks,
                     validation_data=(x_valid, y_valid),
#                     validation_data=val_gen, 
#                     validation_steps = math.ceil(2468645 / batch_size), 
                     max_q_size=10,
                     workers = 4,
                     verbose = VERBOSEFLAG 
                  )







test_df = pd.read_csv("../input/test_simplified.csv")

n_samples = test_df.shape[0]
pick_per_epoch = math.ceil(n_samples / batch_size)
pick_order = np.arange(test_df.shape[0])

all_preds = []

for i in trange(pick_per_epoch):
        c_pick = pick_order[i*batch_size: (i+1)*batch_size]
        dfs = test_df.iloc[c_pick]
        out_imgs = list(map(strokes_to_img, dfs["drawing"]))
        X = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
        preds = model.predict(X)
        for x in preds:
            all_preds.append(to_class[np.argmax(x)])
#        if i == 50:  # TODO: let it run till completion
#            break

#test_predictions = model.predict(x_test, batch_size=128, verbose=1)
#
#top3 = preds2catids(test_predictions)
#top3.head()
#top3.shape
#
#cats = list_all_categories()
#id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
#top3cats = top3.replace(id2cat)
#top3cats.head()
#top3cats.shape
#
#test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
#submission = test[['key_id', 'word']]
#submission.to_csv(inDir+'/submissions/Prav_simple_model_submission_{}.csv'.format(int(map3 * 10**4)), index=False)
#submission.head()
#submission.shape
        
fdf = pd.DataFrame({"key_id": test_df["key_id"], "word": all_preds + ([""] * (test_df.shape[0] - len(all_preds)))})  # TODO: No need to kill it early
fdf.to_csv("mobilenet_submit.csv", index=False)





