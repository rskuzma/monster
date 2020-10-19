# create doc2vec model of monster jobs to calculate similarity

### imports
import pandas as pd
import pickle
# import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
# from gensim.utils import simple_preprocess

# from nltk.corpus import stopwords
# STOPWORDS = stopwords.words('english')



class TaggedDocumentIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc, tags=[self.labels_list[idx]])



### load data
CLEAN_DATA_PATH = '/Users/richardkuzma/coding/analysis/monster/data/cleaned/'
# load cleaned text
with open(CLEAN_DATA_PATH + 'monster_jobs_cleaned_text.pkl', 'rb') as f:
    cleaned_text = pickle.load(f)

# load df
with open(CLEAN_DATA_PATH + 'monster_jobs_df_small.pkl', 'rb') as f:
    df = pickle.load(f)
df['cleaned_description'] = cleaned_text
docs = list(df['cleaned_description'])
labels = list(df['id'])

sentences = TaggedDocumentIterator(docs, labels)


# create doc2vec model
model = Doc2Vec(vector_size=100,
                 window=5,
                 min_count=20,
                 workers=1,
                 alpha=0.025,
#                  min_alpha=0.0025,
                 epochs=100,
                 dm=0,        #1 = paragraph vector - distributed memory; 0 = dbow
                 seed=42)

# train model
model.build_vocab(sentences)
model.train(sentences,
             total_examples=model.corpus_count,
             epochs=model.epochs)

print('created Doc2Vec model')
print('Distributed Mem (True) or BOW (false): {}'.format(model.dm))
print('vector length: {}'.format(model.vector_size))
print('corpus count: {}'.format(model.corpus_count))
print('epochs: {}'.format(model.epochs))


print('saving d2v model...')
MODEL_PATH = '/Users/richardkuzma/coding/analysis/monster/models/'
model_name = 'd2v_' + 'dm_' + str(int(model.dm)) +'_vecsize_' + str(model.vector_size) + '_epochs_' + str(model.epochs) + '.model'
model.save(MODEL_PATH + model_name)
print('saved d2v model. \n location: ' + MODEL_PATH + '\nname: ' + model_name)
