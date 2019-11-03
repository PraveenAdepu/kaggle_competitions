from __future__ import print_function
import numpy as np
import pandas as pd
import csv, datetime, time, json
from zipfile import ZipFile
from os.path import expanduser, exists
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Activation
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.data_utils import get_file
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers.recurrent import LSTM, GRU


inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
#train_df = pd.read_csv(train_file)
#test_df = pd.read_csv(test_file)
#print(train_df.shape) # (404290, 6)
#print(test_df.shape)  # (2345796, 3)

#data = train_df
#y = train_df['is_duplicate']

KERAS_DATASETS_DIR = inDir

QUESTION_PAIRS_FILE = train_file

GLOVE_FILE = 'C:/Users/SriPrav/Documents/R/23Quora/preModels/glove.840B.300d.txt'

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 300
MODEL_WEIGHTS_FILE = 'dn6_question_pairs_weights.h5'
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
RNG_SEED = 2017
NB_EPOCHS = 20

TRAIN_LENGTH = 0


print("Processing", QUESTION_PAIRS_FILE)

question1 = []
question2 = []
is_duplicate = []
with open(QUESTION_PAIRS_FILE) as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        question1.append(row['question1'])
        question2.append(row['question2'])
        is_duplicate.append(row['is_duplicate'])

print('Question pairs: %d' % len(question1))
TRAIN_LENGTH = len(question1)

print('preprocessing testfile')
test_pd = pd.read_csv(test_file)
question1_ = list(test_pd['question1'])
question2_ = list(test_pd['question2'])
for i in range(len(question1_)):
    if type(question1_[i]) != str:
        question1_[i] = 'nan'
for i in range(len(question2_)):
    if type(question2_[i]) != str:
        question2_[i] = 'nan'

question1 += question1_
question2 += question2_

questions = question1 + question2
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)
question1_word_sequences = tokenizer.texts_to_sequences(question1)
question2_word_sequences = tokenizer.texts_to_sequences(question2)
word_index = tokenizer.word_index

print("Words in index: %d" % len(word_index))



print("Processing", GLOVE_FILE)

embeddings_index = {}
with open(GLOVE_FILE) as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings: %d' % len(embeddings_index))

nb_words = min(MAX_NB_WORDS, len(word_index))
word_embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        word_embedding_matrix[i] = embedding_vector
    
print('Null word embeddings: %d' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))

q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(is_duplicate, dtype=int)
print('Shape of question1 data tensor:', q1_data.shape)
print('Shape of question2 data tensor:', q2_data.shape)
print('Shape of label tensor:', labels.shape)



X = np.stack((q1_data, q2_data), axis=1)
X_t = X[:TRAIN_LENGTH]
X_pred = X[TRAIN_LENGTH:]
y = labels

X_train, X_test, y_train, y_test = train_test_split(X_t, y, test_size=TEST_SPLIT, random_state=RNG_SEED)
Q1_train = X_train[:,0]
Q2_train = X_train[:,1]
Q1_test = X_test[:,0]
Q2_test = X_test[:,1]
Q1_pred = X_pred[:,0]
Q2_pred = X_pred[:,1]

Q1 = Sequential()
Q1.add(Embedding(nb_words + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, dropout=0.1))#, dropout=0.2
#Q1.add(TimeDistributed(Dense(EMBEDDING_DIM)))
Q1.add(LSTM(EMBEDDING_DIM, dropout_W=0.2, dropout_U=0.2))
#Q1.add(Activation('relu'))
#Q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
Q2 = Sequential()
Q2.add(Embedding(nb_words + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, dropout=0.1))#, dropout=0.2
#Q2.add(TimeDistributed(Dense(EMBEDDING_DIM)))
Q2.add(LSTM(EMBEDDING_DIM, dropout_W=0.2, dropout_U=0.2))
#Q2.add(Activation('relu'))
#Q2.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, )))
model = Sequential()
model.add(Merge([Q1, Q2], mode='concat'))
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

callbacks = [ModelCheckpoint(MODEL_WEIGHTS_FILE, monitor='val_loss', save_best_only=True)]

print("Starting training at", datetime.datetime.now())

t0 = time.time()
history = model.fit([Q1_train, Q2_train], 
                    y_train, 
                    nb_epoch=NB_EPOCHS, 
                    validation_split=VALIDATION_SPLIT, 
                    verbose=2, 
                    callbacks=callbacks)
t1 = time.time()

print("Training ended at", datetime.datetime.now())

print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

model.load_weights('C:/Users/SriPrav/Documents/R/23Quora/dn6_question_pairs_weights.h5')

#loss, accuracy = model.evaluate([Q1_test, Q2_test], y_test)
#print('')
#print('loss      = {0:.4f}'.format(loss))
#print('accuracy  = {0:.4f}'.format(accuracy))
#print('precision = {0:.4f}'.format(precision))
#print('recall    = {0:.4f}'.format(recall))
#print('F         = {0:.4f}'.format(fbeta_score))

preds = model.predict([Q1_pred, Q2_pred])
sub = pd.DataFrame()
test_pd = pd.read_csv(test_file)
sub['test_id'] = test_pd['test_id']
sub['is_duplicate'] = preds

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/submissions/Prav.deepnet06.csv'
sub.to_csv(sub_file, index=False)
