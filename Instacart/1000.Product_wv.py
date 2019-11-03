# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 20:12:38 2017

@author: SriPrav
"""

import pandas as pd
import numpy as np
import gensim
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

%matplotlib inline

inDir = 'C:/Users/SriPrav/Documents/R/26Instacart'

train_orders_file  = inDir + "/input/order_products__train.csv"
prior_orders_file  = inDir + "/input/order_products__prior.csv"
products_file = inDir + "/input/products.csv"
train_orders = pd.read_csv(train_orders_file)
prior_orders = pd.read_csv(prior_orders_file)
products = pd.read_csv(products_file)

print(train_orders.shape) # (40479, 4)
print(prior_orders.shape)  # (40669, 2)
print(products.shape)  # (20522, 2)

train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

sentences = prior_products.append(train_products).values

model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

vocab = list(model.wv.vocab.keys())

def get_batch(vocab, model, n_batches=3):
    output = list()
    for i in range(0, n_batches):
        rand_int = np.random.randint(len(vocab), size=1)[0]
        suggestions = model.most_similar(positive=[vocab[rand_int]], topn=5)
        suggest = list()
        for i in suggestions:
            suggest.append(i[0])
        output += suggest
        output.append(vocab[rand_int])
    return output

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    """From Tensorflow's tutorial."""
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
#     plt.savefig(filename)
    plt.show()
    
tsne = TSNE()
embeds = []
labels = []

for item in vocab:
    embeds.append(model[item])
embeds = np.array(embeds)    
embeds = tsne.fit_transform(embeds) 

products_id = pd.DataFrame(vocab) 
embed_ids = pd.DataFrame(embeds)
products_id.columns = ['product_id']
embed_ids.columns = ['v1','v2']
product_embeds =pd.concat([products_id, embed_ids], axis=1)

sub_file = inDir +'/input/product_vector_features2.csv'
product_embeds.to_csv(sub_file, index=False)
    
#for item in get_batch(vocab, model, n_batches=3000):
#    embeds.append(model[item])
#    labels.append(products.loc[int(item)]['product_name'])
#embeds = np.array(embeds)
#
#embeds = tsne.fit_transform(embeds)
#plot_with_labels(embeds, labels)
#
#model.save("product2vec.model")


