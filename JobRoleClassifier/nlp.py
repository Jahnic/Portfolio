import pandas as pd 
import numpy as np
# Preprocessing
import re
# NLP
# Logisic regression
# BERT 

data = pd.read_csv("data/data_clean.csv")
# Job descriptions 
descriptions = data[['Job Description', 'title']]
descriptions.columns = ['desc', 'title'] 
descriptions.title.value_counts()
print(descriptions.desc[0])

# Binary labels 
descriptions['data_science'] = descriptions.title.apply(
    lambda x: 1 if x == 'Data Science' else 0
)
descriptions['data_engineering'] = descriptions.title.apply(
    lambda x: 1 if x == 'Data Engineering' else 0
)
descriptions['data_analytics'] = descriptions.title.apply(
    lambda x: 1 if x == 'Data Analysis' else 0
)
descriptions['ml_engineering'] = descriptions.title.apply(
    lambda x: 1 if x == 'ML Engineering' else 0
)
descriptions.describe()

# Remove punctuation
descriptions.desc[0]
letters_only = descriptions.desc.apply(
    lambda x: re.sub("[^a-zA-Z]", # non-letters
                      " ", # avoid fusing letters
                      x)
)

lower_case = letters_only.lower()

# Training data
ds_labels = np.array(descriptions)
de_labels
da_labels
ml_labels