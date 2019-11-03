# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 13:58:02 2018

@author: SriPrav
"""

%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import os
import ast
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
#from tensorflow import keras
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import math
#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet50 import preprocess_input

import gc
start = dt.datetime.now()

inDir = r"C:/Users/SriPrav/Documents/R/55Doodle"

DP_DIR = inDir
INPUT_DIR = inDir+'/input'

BASE_SIZE = 256
NCSVS = 100
NCATS = 340
np.random.seed(seed=1987)
tf.set_random_seed(seed=1987)

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


EPOCHS = 1
size = 128
batchsize = 64
learning_rate = 1e-3
STEPS = 400000
valSTEPS = 200#math.ceil(254952/batchsize)

def strokes_to_img(in_strokes):
    #in_strokes = eval(in_strokes)
    # make an agg figure
    fig, ax = plt.subplots()
    for x,y in in_strokes:
        ax.plot(x, y, linewidth=12.) #  marker='.',
    ax.axis('off')
    fig.canvas.draw()
    
    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    return cv2.resize(X, (size, size))
    #return (cv2.resize(X, (size, size)) / 255.)[::-1]

#def train_gen():
#    while True:  # Infinity loop
#        np.random.shuffle(pick_order)
#        for i in range(pick_per_epoch):
#            c_pick = pick_order[i*batch_size: (i+1)*batch_size]
#            dfs = df.iloc[c_pick]
#            out_imgs = list(map(strokes_to_img, dfs["drawing"]))
#            X = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
#            y = np.array([classes[x] for x in dfs["word"]])
#            yield X, y
            
def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator_xd(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                out_imgs = list(map(strokes_to_img, df["drawing"]))
                x = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y

def image_generator_xd_valid(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                out_imgs = list(map(strokes_to_img, df["drawing"]))
                x = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x, y
def image_generator_xd_validpredict(size, batchsize, ks, lw=6, time_color=True):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batchsize):
                df['drawing'] = df['drawing'].apply(ast.literal_eval)
                out_imgs = list(map(strokes_to_img, df["drawing"]))
                x = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.y, num_classes=NCATS)
                yield x
                
def df_to_image_array_xd(df, size, lw=6, time_color=True):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    out_imgs = list(map(strokes_to_img, df["drawing"]))
    x = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
#    x = np.zeros((len(df), size, size, 1))
#    for i, raw_strokes in enumerate(df.drawing.values):
#        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    x = preprocess_input(x).astype(np.float32)
    return x


train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))
valid_datagen = image_generator_xd_valid(size=size, batchsize=batchsize, ks=(NCSVS - 1))
valid_datagen_predict = image_generator_xd_validpredict(size=size, batchsize=batchsize, ks=(NCSVS - 1))
x, y = next(train_datagen)

for i in range(batchsize):
    plt.imshow(x[i,:,:,:3])
    plt.show()


MODEL_WEIGHTS_FILE = inDir + '/Prav_Model10.h5'

callbacks = [
    ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=1, mode='max', cooldown=3, verbose=1),
    ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_categorical_accuracy', save_best_only=True, verbose=1),
]

#model = model_ResNet50(NCATS)
model = MobileNet(input_shape=(size, size, 3), alpha=1., weights=None, classes=NCATS)
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy',
              metrics=[categorical_crossentropy, categorical_accuracy, top_3_accuracy])
print(model.summary())

model.fit_generator(
                        train_datagen, 
                        steps_per_epoch=STEPS, 
#                        initial_epoch = initial_epoch,
                        epochs=EPOCHS,
                        verbose=1,
                        validation_data=valid_datagen , #(x_valid, y_valid),
                        validation_steps = valSTEPS,
                        max_q_size=10,
                        callbacks = callbacks
)


model.load_weights(MODEL_WEIGHTS_FILE)

k = 100

test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
test1 = test[:10000]
x_test = df_to_image_array_xd(test1,size=size)
test_predictions1 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()
    
test2 = test[10000:20000]
x_test = df_to_image_array_xd(test2,size=size)
test_predictions2 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()
        
test3 = test[20000:30000]
x_test = df_to_image_array_xd(test3,size=size)
test_predictions3 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()

test4 = test[30000:40000]
x_test = df_to_image_array_xd(test4,size=size)
test_predictions4 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()

test5 = test[40000:50000]
x_test = df_to_image_array_xd(test5,size=size)
test_predictions5 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()

test6 = test[50000:60000]
x_test = df_to_image_array_xd(test6,size=size)
test_predictions6 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()

test7 = test[60000:70000]
x_test = df_to_image_array_xd(test7,size=size)
test_predictions7 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()

test8 = test[70000:80000]
x_test = df_to_image_array_xd(test8,size=size)
test_predictions8 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()

test9 = test[80000:90000]
x_test = df_to_image_array_xd(test9,size=size)
test_predictions9 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()

test10 = test[90000:100000]
x_test = df_to_image_array_xd(test10,size=size)
test_predictions10 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()

test11 = test[110000:]
x_test = df_to_image_array_xd(test11,size=size)
test_predictions11 = model.predict(x_test, batch_size=128, verbose=1)
if (k == 100):
    gc.collect()


top3 = preds2catids(test_predictions)
top3.head()
top3.shape

cats = list_all_categories()
id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
top3cats = top3.replace(id2cat)
top3cats.head()
top3cats.shape

test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
submission = test[['key_id', 'word']]
submission.to_csv(inDir+'/submissions/Prav_simple_model_submission_75k_64_{}.csv'.format(int(map3 * 10**4)), index=False)
submission.head()
submission.shape

end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))