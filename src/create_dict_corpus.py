# Preprocess Monster.com Jobs
# - Richard Kuzma, 13OCT2020
# - Pulled 22,000 Monster.com jobs in a csv from kaggle

### imports
import numpy as np
import pandas as pd
import pickle

# import custom module preprocess_helpers
import os
import sys
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import utils.preprocess_helpers as ph


# Load df
CLEAN_DATA_PATH = '/Users/richardkuzma/coding/analysis/monster/data/cleaned/'
df_filename = 'monster_jobs_df_small.pkl'
with open(CLEAN_DATA_PATH+df_filename, 'rb') as f:
    df = pickle.load(f)
print('Loaded df')

# clean data
jobs_raw = df['job_description'].tolist()
jobs_cleaned = ph.full_clean(jobs_raw)

# make dictionary
jobs_dict = ph.make_dict(jobs_cleaned)
jobs_dict.filter_extremes(no_below=20, no_above=0.5)
jobs_dict.compactify() # remove IDs for removed words

# make corpus
jobs_corpus = ph.make_bow(jobs_cleaned, jobs_dict)

### save cleaned text, dictionary, corpus
jobs_cleaned_filename = 'monster_jobs_cleaned_text.pkl'
with open(CLEAN_DATA_PATH+jobs_cleaned_filename, 'wb') as f:
    pickle.dump(jobs_cleaned, f)
print('Saved cleaned jobs text')

jobs_dict_filename = 'monster_jobs_dict.pkl'
with open(CLEAN_DATA_PATH+jobs_dict_filename, 'wb') as f:
    pickle.dump(jobs_dict, f)
print('Saved jobs dictionary')

jobs_corpus_filename = 'monster_jobs_corpus.pkl'
with open(CLEAN_DATA_PATH+jobs_corpus_filename, 'wb') as f:
    pickle.dump(jobs_corpus, f)
print('Saved jobs corpus')
