
# # Get Topics (Skills), Words (sub-skills)
# - Richard Kuzma, 13OCT2020

# basic
from pprint import pprint
import pickle
import time
# data science
import pandas as pd
import numpy as np
print(np.__version__)
# NLP
import gensim
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
# from gensim.test.utils import datapath
# plotting
import pyLDAvis
import pyLDAvis.gensim
# import matplotlib.pyplot as plt
### Import preprocesss_helpers module
import os
import sys
# add '/Users/richardkuzma/coding/analysis' to path
# where /utils holds module preprocess_helpers
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils.preprocess_helpers as ph

### methods
def gather_words_from_LDA_topics(model):
    """list of length model.n_topics with top words_per_topic words associated with each topic"""
    topic_words = []
    total_missed_words = 0
    words_per_topic = 100
    for i in range (0, model.num_topics): # for each LDA topic
        missed_words = 0
        this_topic_words = []
        for j in range(0, words_per_topic): # for first 'words_per_topic' words in a topic
            try:
                this_topic_words.append(model.show_topic(i, topn=words_per_topic)[j][0])
            except KeyError:
                missed_words +=1

        total_missed_words += missed_words
        topic_words.append(this_topic_words)
    return topic_words

### generate 'skill' topics for document, with weights
### note, not all topics may be represented
def get_doc_topics(document_string: str, model):
    # turn string into list of length 1
    doc_string = [document_string]
    # clean text into lemmatized lists of unigrams, bigrams, trigrams
    doc_cleaned = ph.full_clean(doc_string)
    # create dictionary
    doc_dict = ph.make_dict(doc_cleaned)
    # make corpus
    doc_bow = ph.make_bow(doc_cleaned, doc_dict)
    # create lda probability distribution of topics
    # doc_lda is a gensim transformed corpus
    doc_lda = model[doc_bow]

    # only one document means only one element, a list of tuples with (topic, probability)
    for topic in doc_lda:
        doc_topics = topic
        # there should only be one list, so break
        break

    # list of tuples
    return doc_topics

# order top 'skills' of a document
def get_doc_top_topics(doc_topic_list: list):
    doc_topic_list.sort(reverse=True, key=lambda x: x[1])
    ordered_topic_list = []
    for i in range(0, len(doc_topic_list)):
        ordered_topic_list.append(doc_topic_list[i][0])
    return ordered_topic_list

#     e.g. input:  [(6, 0.09539536), (8, 0.045605347), (24, 0.23120484), (32, 0.19147022), (34, 0.40980446)]
#          output: [34, 24, 32, 6, 8]

### Load corpus and dict
CLEAN_DATA_PATH = '/Users/richardkuzma/coding/analysis/monster/data/cleaned/'
MODEL_PATH = '/Users/richardkuzma/coding/analysis/monster/models/'

# jobs_cleaned_filename = 'monster_jobs_cleaned_text.pkl'
# with open(path+jobs_cleaned_filename, 'rb') as f:
#     jobs_cleaned = pickle.load(f)

# load dictionary
jobs_dict_filename = 'monster_jobs_dict.pkl'
with open(CLEAN_DATA_PATH+jobs_dict_filename, 'rb') as f:
    jobs_dict = pickle.load(f)

# load corpus
jobs_corpus_filename = 'monster_jobs_corpus.pkl'
with open(CLEAN_DATA_PATH+jobs_corpus_filename, 'rb') as f:
    jobs_corpus = pickle.load(f)

# Load LDA model
model_name = 'LDA_20_topics.model'
model = LdaModel.load(MODEL_PATH+model_name)
print('loaded LDA model')
pprint(model.print_topics())

# load df
df_filename = 'monster_jobs_df_small.pkl'
with open(CLEAN_DATA_PATH+df_filename, 'rb') as f:
    df = pickle.load(f)

print('loaded dictionary, corpus, LDA model, df')


# identify topics for each job post document
# this took >35min
print('getting document topics...')
start_time = time.time()
df['doc_topics'] = df.apply(lambda x: get_doc_topics(x['job_description'], model), axis=1)
print('finished. {} seconds'.format(time.time()-start_time))

# order topics
print('ordering document topics...')
start_time = time.time()
df['doc_top_topics'] = df.apply(lambda x: get_doc_top_topics(x['doc_topics']), axis=1)
print('finished. {} seconds'.format(time.time()-start_time))

### save df
df_filename = 'monster_jobs_df_with_topics.pkl'
with open(CLEAN_DATA_PATH+df_filename, 'wb') as f:
    pickle.dump(df, f)
print('saved df with topics')

print('getting top words for each topic...')
start_time = time.time()
topic_words = gather_words_from_LDA_topics(model)
print('finished. {} seconds'.format(time.time()-start_time))


topic_words_filename = '20_topics_100_words_each.pkl'
with open(CLEAN_DATA_PATH+topic_words_filename, 'wb') as f:
    pickle.dump(topic_words, f)
print('saved top words for each topic')
