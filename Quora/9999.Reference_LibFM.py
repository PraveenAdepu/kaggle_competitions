# -*- coding: utf-8 -*-
"""
Created on Wed May 03 10:41:59 2017

@author: PAdepu
"""

import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
import xgboost as xgb
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from nltk.corpus import stopwords
from collections import Counter

stops = set(stopwords.words("english"))

inDir = 'C:/Users/padepu/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)
#################################################################################################################
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
seed = 1024
np.random.seed(seed)

inDir = 'C:/Users/padepu/Documents/R/23Quora'

train = pd.read_csv(inDir+"/input/train.csv")
test = pd.read_csv(inDir+"/input/test.csv")

def stem_str(x,stemmer=SnowballStemmer('english')):
    x = text.re.sub("[^a-zA-Z0-9]"," ", x)
    x = (" ").join([stemmer.stem(z) for z in x.split(" ")])
    x = " ".join(x.split())
    return x

porter = PorterStemmer()
snowball = SnowballStemmer('english')


print('Generate porter')
train['question1_porter'] = train['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question1_porter'] = test['question1'].astype(str).apply(lambda x:stem_str(x.lower(),porter))

train['question2_porter'] = train['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))
test['question2_porter'] = test['question2'].astype(str).apply(lambda x:stem_str(x.lower(),porter))

train.to_csv(inDir+'/input/train_porter.csv')
test.to_csv(inDir+'/input/test_porter.csv')

################################################################################################################

import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
seed = 1024
np.random.seed(seed)
inDir = 'C:/Users/padepu/Documents/R/23Quora'

ft = ['question1','question2','question1_porter','question2_porter']
train = pd.read_csv(inDir+"/input/train_porter.csv")[ft]
test = pd.read_csv(inDir+"/input/test_porter.csv")[ft]
# test['is_duplicated']=[-1]*test.shape[0]

len_train = train.shape[0]

data_all = pd.concat([train,test])
print data_all

max_features = None
ngram_range = (1,2)
min_df = 3
print('Generate tfidf')
feats= ['question1','question2']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus = []
for f in feats:
    data_all[f] = data_all[f].astype(str)
    corpus+=data_all[f].values.tolist()

vect_orig.fit(
    corpus
    )

for f in feats:
    tfidfs = vect_orig.transform(data_all[f].values.tolist())
    train_tfidf = tfidfs[:train.shape[0]]
    test_tfidf = tfidfs[train.shape[0]:]
    pd.to_pickle(train_tfidf,inDir+'/input/train_%s_tfidf.pkl'%f)
    pd.to_pickle(test_tfidf,inDir+'/input/test_%s_tfidf.pkl'%f)


print('Generate porter tfidf')
feats= ['question1_porter','question2_porter']
vect_orig = TfidfVectorizer(max_features=max_features,ngram_range=ngram_range, min_df=min_df)

corpus = []
for f in feats:
    data_all[f] = data_all[f].astype(str)
    corpus+=data_all[f].values.tolist()

vect_orig.fit(
    corpus
    )

for f in feats:
    tfidfs = vect_orig.transform(data_all[f].values.tolist())
    train_tfidf = tfidfs[:train.shape[0]]
    test_tfidf = tfidfs[train.shape[0]:]
    pd.to_pickle(train_tfidf,inDir+'/input/train_%s_tfidf.pkl'%f)
    pd.to_pickle(test_tfidf,inDir+'/input/test_%s_tfidf.pkl'%f)

#######################################################################################################

import pandas as pd
import numpy as np
from scipy import sparse as ssp
from sklearn.model_selection import KFold
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from sklearn.utils import resample,shuffle
from sklearn.preprocessing import MinMaxScaler
seed=1024
np.random.seed(seed)

inDir = 'C:/Users/padepu/Documents/R/23Quora'

train = pd.read_csv(inDir+"/input/train_porter.csv")


# tfidf
train_question1_tfidf = pd.read_pickle(inDir+'/input/train_question1_tfidf.pkl')[:]
test_question1_tfidf = pd.read_pickle(inDir+'/input/test_question1_tfidf.pkl')[:]

train_question2_tfidf = pd.read_pickle(inDir+'/input/train_question2_tfidf.pkl')[:]
test_question2_tfidf = pd.read_pickle(inDir+'/input/test_question2_tfidf.pkl')[:]


train_question1_porter_tfidf = pd.read_pickle(inDir+'/input/train_question1_porter_tfidf.pkl')[:]
test_question1_porter_tfidf = pd.read_pickle(inDir+'/input/test_question1_porter_tfidf.pkl')[:]

train_question2_porter_tfidf = pd.read_pickle(inDir+'/input/train_question2_porter_tfidf.pkl')[:]
test_question2_porter_tfidf = pd.read_pickle(inDir+'/input/test_question2_porter_tfidf.pkl')[:]

train.head(1)

#train_interaction = pd.read_pickle(path+'train_interaction.pkl')[:].reshape(-1,1)
#test_interaction = pd.read_pickle(path+'test_interaction.pkl')[:].reshape(-1,1)
#
#train_porter_interaction = pd.read_pickle(path+'train_porter_interaction.pkl')[:].reshape(-1,1)
#test_porter_interaction = pd.read_pickle(path+'test_porter_interaction.pkl')[:].reshape(-1,1)
#
#
#train_jaccard = pd.read_pickle(path+'train_jaccard.pkl')[:].reshape(-1,1)
#test_jaccard = pd.read_pickle(path+'test_jaccard.pkl')[:].reshape(-1,1)
#
#train_porter_jaccard = pd.read_pickle(path+'train_porter_jaccard.pkl')[:].reshape(-1,1)
#test_porter_jaccard = pd.read_pickle(path+'test_porter_jaccard.pkl')[:].reshape(-1,1)

#train_len = pd.read_pickle(path+"train_len.pkl")
#test_len = pd.read_pickle(path+"test_len.pkl")
#scaler = MinMaxScaler()
#scaler.fit(np.vstack([train_len,test_len]))
#train_len = scaler.transform(train_len)
#test_len =scaler.transform(test_len)


X = ssp.hstack([
    train_question1_tfidf,
    train_question2_tfidf,
#    train_interaction,
#    train_porter_interaction,
#    train_jaccard,
#    train_porter_jaccard,
#    train_len,
    ]).tocsr()


y = train['is_duplicate'].values[:]

X_t = ssp.hstack([
    test_question1_tfidf,
    test_question2_tfidf,
#    test_interaction,
#    test_porter_interaction,
#    test_jaccard,
#    test_porter_jaccard,
#    test_len,
    ]).tocsr()


print X.shape
print X_t.shape

skf = KFold(n_splits=5, shuffle=True, random_state=seed).split(X)
for ind_tr, ind_te in skf:
    X_train = X[ind_tr]
    X_test = X[ind_te]

    y_train = y[ind_tr]
    y_test = y[ind_te]
    break

dump_svmlight_file(X,y,inDir+"/input/X_tfidf.svm")
del X
dump_svmlight_file(X_t,np.zeros(X_t.shape[0]),inDir+"/input/X_t_tfidf.svm")
del X_t

def oversample(X_ot,y,p=0.165):
    pos_ot = X_ot[y==1]
    neg_ot = X_ot[y==0]
    #p = 0.165
    scale = ((pos_ot.shape[0]*1.0 / (pos_ot.shape[0] + neg_ot.shape[0])) / p) - 1
    while scale > 1:
        neg_ot = ssp.vstack([neg_ot, neg_ot]).tocsr()
        scale -=1
    neg_ot = ssp.vstack([neg_ot, neg_ot[:int(scale * neg_ot.shape[0])]]).tocsr()
    ot = ssp.vstack([pos_ot, neg_ot]).tocsr()
    y=np.zeros(ot.shape[0])
    y[:pos_ot.shape[0]]=1.0
    print y.mean()
    return ot,y

X_train,y_train = oversample(X_train.tocsr(),y_train,p=0.165)
X_test,y_test = oversample(X_test.tocsr(),y_test,p=0.165)

X_train,y_train = shuffle(X_train,y_train,random_state=seed)

dump_svmlight_file(X_train,y_train,inDir+"/input/X_train_tfidf.svm")
dump_svmlight_file(X_test,y_test,inDir+"/input/X_test_tfidf.svm")

###########################################################################################################

#############################################################################################################
# Created by qqgeogor
# https://www.kaggle.com/qqgeogor
#############################################################################################################

from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt,pow
import itertools
import math
from random import random,shuffle,uniform,seed
import pickle
import sys

seed(1024)

def data_generator(path,no_norm=False,task='c'):
    data = open(path,'r')
    for row in data:
        row = row.strip().split(" ")
        y = float(row[0])
        row = row[1:]
        x = []
        for feature in row:
            feature = feature.split(":")
            idx = int(feature[0])
            value = float(feature[1])
            x.append([idx,value])

        if not no_norm:
            r = 0.0
            for i in range(len(x)):
                r+=x[i][1]*x[i][1]
            for i in range(len(x)):
                x[i][1] /=r
        # if task=='c':
        #     if y ==0.0:
        #         y = -1.0

        yield x,y


def dot(u,v):
    u_v = 0.
    len_u = len(u)
    for idx in range(len_u):
        uu = u[idx]
        vv = v[idx]
        u_v+=uu*vv
    return u_v

def mse_loss_function(y,p):
    return (y - p)**2

def mae_loss_function(y,p):
    y = exp(y)
    p = exp(p)
    return abs(y - p)

def log_loss_function(y,p):
    return -(y*log(p)+(1-y)*log(1-p))

def exponential_loss_function(y,p):
    return log(1+exp(-y*p))

def sigmoid(inX):
    return 1/(1+exp(-inX))

def bounded_sigmoid(inX):
    return 1. / (1. + exp(-max(min(inX, 35.), -35.)))


class SGD(object):
    def __init__(self,lr=0.001,momentum=0.9,nesterov=True,adam=False,l2=0.0,l2_fm=0.0,l2_bias=0.0,ini_stdev= 0.01,dropout=0.5,task='c',n_components=4,nb_epoch=5,interaction=False,no_norm=False):
        self.W = []
        self.V = []        
        self.bias = uniform(-ini_stdev, ini_stdev)
        self.n_components=n_components
        self.lr = lr
        self.l2 = l2
        self.l2_fm = l2_fm
        self.l2_bias = l2_bias
        self.momentum = momentum
        self.nesterov = nesterov
        self.adam = adam
        self.nb_epoch = nb_epoch
        self.ini_stdev = ini_stdev
        self.task = task
        self.interaction = interaction
        self.dropout = dropout
        self.no_norm = no_norm
        if self.task!='c':
            # self.loss_function = mse_loss_function
            self.loss_function = mae_loss_function
        else:
            # self.loss_function = exponential_loss_function
            self.loss_function = log_loss_function

    def preload(self,train,test):
        train = data_generator(train,self.no_norm,self.task)
        dim = 0
        count = 0
        for x,y in train:
            for i in x:
                idx,value = i
                if idx >dim:
                    dim = idx
            count+=1
        print('Training samples:',count)
        test = data_generator(test,self.no_norm,self.task)
        count=0
        for x,y in test:
            for i in x:
                idx,value = i
                if idx >dim:
                    dim = idx
            count+=1
        print('Testing samples:',count)
        
        dim = dim+1
        print("Number of features:",dim)
        
        self.W = [uniform(-self.ini_stdev, self.ini_stdev) for _ in range(dim)]
        self.Velocity_W = [0.0 for _ in range(dim)]
        
        
        self.V = [[uniform(-self.ini_stdev, self.ini_stdev) for _ in range(self.n_components)] for _ in range(dim)]
        self.Velocity_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]
        
        self.Velocity_bias = 0.0
        
        self.dim = dim
        
        
    def adam_init(self):
        self.iterations = 0
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon=1e-8
        self.decay = 0.
        self.inital_decay = self.decay 

        dim =self.dim

        self.m_W = [0.0 for _ in range(dim)]
        self.v_W = [0.0 for _ in range(dim)]

        self.m_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]
        self.v_V = [[0.0 for _ in range(self.n_components)] for _ in range(dim)]

        self.m_bias = 0.0
        self.v_bias = 0.0


    def adam_update(self,lr,x,residual):

        if 0.<self.dropout<1.:
            self.droupout_x(x)
        
        lr = self.lr
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1

        lr_t = lr * sqrt(1. - pow(self.beta_2, t)) / (1. - pow(self.beta_1, t))
        
        for sample in x:
            idx,value = sample
            g = residual*value

            m = self.m_W[idx]
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v = self.v_W[idx]
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * (g**2)

            p = self.W[idx]
            p_t = p - lr_t *m_t / (sqrt(v_t) + self.epsilon)

            if self.l2>0:
                p_t = p_t - lr_t*self.l2*p

            self.m_W[idx] = m_t
            self.v_W[idx] = v_t
            self.W[idx] = p_t

        if self.interaction:
            self._adam_update_fm(lr_t,x,residual)


        m = self.m_bias
        m_t = (self.beta_1 * m) + (1. - self.beta_1)*residual

        v = self.v_bias
        v_t = (self.beta_2 * v) + (1. - self.beta_2)*(residual**2)

        p = self.bias
        p_t = p - lr_t * m_t / (sqrt(v_t) + self.epsilon)
        if self.l2_bias>0:
            pt = pt - lr_t * self.l2_bias*p

        self.m_bias = m_t
        self.v_bias = v_t
        self.bias = p_t

        self.iterations+=1

    def _adam_update_fm(self,lr_t,x,residual):
        len_x = len(x)
        sum_f_dict = self.sum_f_dict
        n_components = self.n_components
        for f in range(n_components):
            for i in range(len_x):
                idx_i,value_i = x[i]
                v = self.V[idx_i][f]
                sum_f = sum_f_dict[f]
                g = (sum_f*value_i - v *value_i*value_i)*residual

                m = self.m_V[idx_i][f]
                m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

                v = self.v_V[idx_i][f]
                v_t = (self.beta_2 * v) + (1. - self.beta_2) * (g**2)

                p = self.V[idx_i][f]
                p_t = p - lr_t * m_t / (sqrt(v_t) + self.epsilon)

                if self.l2_fm>0:
                    p_t = p_t - lr_t * self.l2_fm*p

                self.m_V[idx_i][f] = m_t
                self.v_V[idx_i][f] = v_t
                self.V[idx_i][f] = p_t

    def droupout_x(self,x):
        new_x = []
        for i, var in enumerate(x):
            if random() > self.dropout:
                del x[i]

    def _predict_fm(self,x):
        len_x = len(x)
        n_components = self.n_components
        pred = 0.0
        self.sum_f_dict = {}
        for f in range(n_components):
            sum_f = 0.0
            sum_sqr_f = 0.0
            for i in range(len_x):
                idx_i,value_i = x[i]
                d = self.V[idx_i][f] * value_i
                sum_f +=d
                sum_sqr_f +=d*d
            pred+= 0.5 * (sum_f*sum_f - sum_sqr_f);
            self.sum_f_dict[f] = sum_f
        return pred

    def _predict_one(self,x):
        pred = self.bias
        # pred = 0.0
        for idx,value in x:
            pred+=self.W[idx]*value
        
        if self.interaction:
            pred+=self._predict_fm(x)

        if self.task=='c':
            pred = bounded_sigmoid(pred)
        return pred


    def _update_fm(self,lr,x,residual):
        len_x = len(x)
        sum_f_dict = self.sum_f_dict
        n_components = self.n_components
        for f in range(n_components):
            for i in range(len_x):
                idx_i,value_i = x[i]
                sum_f = sum_f_dict[f]
                v = self.V[idx_i][f]
                grad = (sum_f*value_i - v *value_i*value_i)*residual
                
                self.Velocity_V[idx_i][f] = self.momentum * self.Velocity_V[idx_i][f] - lr * grad
                if self.nesterov:
                    self.Velocity_V[idx_i][f] = self.momentum * self.Velocity_V[idx_i][f] - lr * grad
                self.V[idx_i][f] = self.V[idx_i][f] + self.Velocity_V[idx_i][f] - lr*self.l2_fm*self.V[idx_i][f]



    def update(self,lr,x,residual):

        if 0.<self.dropout<1.:
            self.droupout_x(x)

        for sample in x:
            idx,value = sample
            grad = residual*value
            self.Velocity_W[idx] =  self.momentum * self.Velocity_W[idx] - lr * grad
            if self.nesterov:
                 self.Velocity_W[idx] = self.momentum * self.Velocity_W[idx] - lr * grad
            self.W[idx] = self.W[idx] + self.Velocity_W[idx] - lr*self.l2*self.W[idx]
            
        if self.interaction:
            self._update_fm(lr,x,residual)

        self.Velocity_bias = self.momentum*self.Velocity_bias - lr*residual
        if self.nesterov:
            self.Velocity_bias = self.momentum*self.Velocity_bias - lr*residual
        self.bias = self.bias +self.Velocity_bias - lr*self.l2_bias*self.bias

    def predict(self,path,out):

        data = data_generator(path,self.no_norm,self.task)
        y_preds =[]
        with open(out, 'w') as outfile:
            ID = 0
            outfile.write('%s,%s\n' % ('test_id', 'is_duplicate'))
            for d in data:
                x,y = d
                p = self._predict_one(x)
                outfile.write('%s,%s\n' % (ID, str(p)))
                ID+=1


    def validate(self,path):
        data = data_generator(path,self.no_norm,self.task)
        loss = 0.0
        count = 0.0

        for d in data:
            x,y = d
            p = self._predict_one(x)
            loss+=self.loss_function(y,p)
            count+=1
        return loss/count

    def save_weights(self):
        weights = []
        weights.append(self.W)
        weights.append(self.V)
        weights.append(self.bias)
        # weights.append(self.Velocity_W)
        # weights.append(self.Velocity_V)
        weights.append(self.dim)
        pickle.dump(weights,open('sgd_fm.pkl','wb'))

    def load_weights(self):
        weights = pickle.load(open('sgd_fm.pkl','rb'))
        self.W = weights[0]
        self.V = weights[1]
        self.bias = weights[2]
        # self.Velocity_W = weights[3]
        # self.Velocity_V = weights[4]
        self.dim = weights[3]
        

    def train(self,path,valid_path = None,in_memory=False):

        start = datetime.now()
        lr = self.lr
        if self.adam:
            self.adam_init()
            self.update = self.adam_update

        if in_memory:
            data = data_generator(path,self.no_norm,self.task)
            data = [d for d in data]
        best_loss = 999999
        best_epoch = 0
        for epoch in range(1,self.nb_epoch+1):
            if not in_memory:
                data = data_generator(path,self.no_norm,self.task)
            train_loss = 0.0
            train_count = 0
            for x,y in data:
                p = self._predict_one(x)
                if self.task!='c':                    
                    residual = -(y-p)
                else:
                    # residual = -y*(1.0-1.0/(1.0+exp(-y*p)));
                    residual = -(y-p)

                self.update(lr,x,residual)
                if train_count%50000==0:
                    if train_count ==0:
                        print '\ttrain_count: %s, current loss: %.6f'%(train_count,0.0)
                    else:
                        print '\ttrain_count: %s, current loss: %.6f'%(train_count,train_loss/train_count)

                train_loss += self.loss_function(y,p)
                train_count += 1

            epoch_end = datetime.now()
            duration = epoch_end-start
            
            if valid_path:
                valid_loss = self.validate(valid_path)
                print('Epoch: %s, train loss: %.6f, valid loss: %.6f, time: %s'%(epoch,train_loss/train_count,valid_loss,duration))
                if valid_loss<best_loss:
                    best_loss = valid_loss
                    self.save_weights()
                    print 'save_weights'
            else:
                print('Epoch: %s, train loss: %.6f, time: %s'%(epoch,train_loss/train_count,duration))


inDir = 'C:/Users/padepu/Documents/R/23Quora'


sgd = SGD(lr=0.001,adam=True,dropout=0.8,l2=0.00,l2_fm=0.00,task='c',n_components=1,nb_epoch=30,interaction=True,no_norm=False)
sgd.preload(inDir+'/input/X_tfidf.svm',inDir+'/input/X_t_tfidf.svm')
# sgd.load_weights()
sgd.train(inDir+'/input/X_train_tfidf.svm',inDir+'/input/X_test_tfidf.svm',in_memory=False)
sgd.load_weights()
sgd.predict(inDir+'/input/X_test_tfidf.svm',out='valid.csv')
print sgd.validate(inDir+'/input/X_test_tfidf.svm')
sgd.predict(inDir+'/input/X_t_tfidf.svm',out='out.csv')

