import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
# Saves model as file
import pickle
# Verify computational environment
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "99"
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
# Data processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Outlier detection
from sklearn.ensemble import IsolationForest

# Load data
data = pd.read_csv("data/complete_data.csv").iloc[:, 4: ]
data.head()

# Drop redundant data
redundant = ['more_than_$150,000_(%)', '5-person_or_more_households_(%)', 
            'single-parent_families_(%)', 'renters_(%)', 'before_1960_(%)',
            'single-family_homes_(%)', 'university_(%)', 'non-immigrant_population_(%)',
            'french_(%)', 'mr_distance', 'new_area_from_price', 'new_area_from_rooms']
data.drop(redundant, axis=1, inplace=True)

# Drop demographics data
# demographics = ['less_than_$50,000_(%)', 'between_$50,000_and_$80,000_(%)', 
#                 'between_$80,000_and_$100,000_(%)', 'between_$100,000_and_$150,000_(%)',
#                 'more_than_$150,000_(%)', '1-person_households_(%)', 
#                 '2-person_households_(%)', '3-person_households_(%)', 
#                 '4-person_households_(%)', '5-person_or_more_households_(%)', 
#                 'couples_without_children_at_home_(%)', 'couples_with_children_at_home_(%)',
#                 'single-parent_families_(%)', 'owners_(%)', 'renters_(%)',
#                 'before_1960_(%)', 'between_1961_and_1980_(%)',
#                 'between_1981_and_1990_(%)', 'between_1991_and_2000_(%)',
#                 'between_2001_and_2010_(%)', 'between_2011_and_2016_(%)',
#                 'single-family_homes_(%)', 'semi-detached_or_row_houses_(%)',
#                 'buildings_with_less_than_5_floors_(%)',
#                 'buildings_with_5_or_more_floors_(%)', 'mobile_homes_(%)',
#                 'university_(%)', 'college_(%)', 'secondary_(high)_school_(%)',
#                 'apprentice_or_trade_school_diploma_(%)', 'no_diploma_(%)',
#                 'non-immigrant_population_(%)', 'immigrant_population_(%)',
#                 'french_(%)', 'english_(%)', 'others_languages_(%)', 'mr_distance',
#                 'new_area_from_price', 'new_area_from_rooms']

# Random index shuffling for train/test split
df = data.copy().sample(frac=1, random_state=0)
# Prepare train and test data
train_size = round(0.8*df.shape[0])
train = df[: train_size]
test = df[train_size : ]

# Inspect data
print('Shape of the train data with all features:', train.shape)
print("")
print("List of features:")
print(list(train.columns))

# Outlier detection
clf = IsolationForest(max_samples = 100, random_state = 42)
clf.fit(train)
y_noano = clf.predict(train)
y_noano = pd.DataFrame(y_noano, columns = ['Top'])
# Indices of non-outliers
noano_indices = y_noano[y_noano['Top'] == 1].index.values

# Anamolies removed
train_ano_rm = train.iloc[noano_indices]
train.reset_index(drop = True, inplace = True)
print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])
print("Number of rows without outliers:", train_ano_rm.shape[0])