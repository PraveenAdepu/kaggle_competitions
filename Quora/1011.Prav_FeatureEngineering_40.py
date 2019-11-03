import pandas as pd
import hashlib
import gc

inDir = 'C:/Users/SriPrav/Documents/R/23Quora'

train_file = inDir + "/input/train.csv"
test_file = inDir + "/input/test.csv"
train_df = pd.read_csv(train_file)
train_df = train_df.fillna(' ')
test_df = pd.read_csv(test_file)
test_df = test_df.fillna(' ')

print(train_df.shape) # (404290, 6)
print(test_df.shape)  # (2345796, 3)

# Generating a graph of Questions and their neighbors
def generate_qid_graph_table(row):
    hash_key1 = hashlib.md5(row["question1"]).hexdigest()
    hash_key2 = hashlib.md5(row["question2"]).hexdigest()

    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)


qid_graph = {}
print('Apply to train...')
train_df.apply(generate_qid_graph_table, axis=1)
print('Apply to test...')
test_df.apply(generate_qid_graph_table, axis=1)


def pagerank():
    MAX_ITER = 20
    d = 0.85

    # Initializing -- every node gets a uniform value!
    pagerank_dict = {i: 1 / len(qid_graph) for i in qid_graph}
    num_nodes = len(pagerank_dict)

    for iter in range(0, MAX_ITER):

        for node in qid_graph:
            local_pr = 0

            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor] / len(qid_graph[neighbor])

            pagerank_dict[node] = (1 - d) / num_nodes + d * local_pr

    return pagerank_dict

print('Main PR generator...')
pagerank_dict = pagerank()

def get_pagerank_value(row):
    q1 = hashlib.md5(row["question1"]).hexdigest()
    q2 = hashlib.md5(row["question2"]).hexdigest()
    s = pd.Series({
        "q1_pr": pagerank_dict[q1],
        "q2_pr": pagerank_dict[q2]
    })
    return s

print('Apply to train...')
pagerank_feats_train = train_df.apply(get_pagerank_value, axis=1)
print('Writing train...')
pagerank_feats_train.to_csv("pagerank_train.csv", index=False)
del train_df
gc.collect()
print('Apply to test...')
pagerank_feats_test = test_df.apply(get_pagerank_value, axis=1)
print('Writing test...')
pagerank_feats_test.to_csv("pagerank_test.csv", index=False)

train = pd.concat([train_df, pagerank_feats_train], axis=1) 
test = pd.concat([test_df, pagerank_feats_test], axis=1) 

train_feat = train[['id','q1_pr','q2_pr']]
test_feat = test[['test_id','q1_pr','q2_pr']]

train_feat.head(15)
sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/train_features_40.csv' 
train_feat.to_csv(sub_file, index=False)  

sub_file = 'C:/Users/SriPrav/Documents/R/23Quora/input/test_features_40.csv' 
test_feat.to_csv(sub_file, index=False)  
