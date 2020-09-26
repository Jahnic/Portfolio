import pandas as pd 
import numpy as np
# Preprocessing
import re
# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
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

def preprocess_words(job_description, stop_words):
    # Remove punctuation
    letters_only = re.sub("[^a-zA-Z]", # non-letters
                        " ", # avoid fusing letters
                        job_description)

    # Split into list of lower case words
    lower_case = letters_only.lower().split()
    
    # Remove stop words
    words = [w for w in lower_case if w not in stop_words]
    
    return (" ".join(words))
    
stops = set(stopwords.words('english'))
# Clean job descriptions
descriptions['desc'] = descriptions.desc.apply(
    lambda x: preprocess_words(x, stops)
)



# Training data
ds_labels = np.array(descriptions)
de_labels
da_labels
ml_labels