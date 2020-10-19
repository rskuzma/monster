# Create LDA model of Monster.com Jobs
# - Richard Kuzma, 14OCT2020
# - Pulled 22,000 Monster.com jobs in a csv from kaggle
import numpy as np
import pandas as pd
import pickle
import gensim
from gensim.models import LdaModel

# ## Load cleaned text, dictionary, corpus
CLEAN_DATA_PATH = '/Users/richardkuzma/coding/analysis/monster/data/cleaned/'
jobs_cleaned_filename = 'monster_jobs_cleaned_text.pkl'
with open(CLEAN_DATA_PATH+jobs_cleaned_filename, 'rb') as f:
    jobs_cleaned = pickle.load(f)

jobs_dict_filename = 'monster_jobs_dict.pkl'
with open(CLEAN_DATA_PATH+jobs_dict_filename, 'rb') as f:
    dictionary = pickle.load(f)

jobs_corpus_filename = 'monster_jobs_corpus.pkl'
with open(CLEAN_DATA_PATH+jobs_corpus_filename, 'rb') as f:
    jobs_corpus = pickle.load(f)

"""Select number of topics"""
num_topics = 20


### make and save model
print('Making LDA model with np version {}'.format(np.__version__))
model = LdaModel(corpus=jobs_corpus, num_topics=num_topics, id2word=dictionary)

print('Saving model..')
MODEL_PATH = '/Users/richardkuzma/coding/analysis/monster/models/'
filename = 'LDA_' + str(num_topics) + '_topics.model'
model.save(MODEL_PATH+filename)
print('Saved model.\nPath: ' + MODEL_PATH + '\nname: ' + filename)
