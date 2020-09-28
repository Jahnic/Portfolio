import pandas as pd 
import numpy as np
# Preprocessing
import re
# Bag of words
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Word2Vec
import nltk.data
import logging
from gensim.models import word2vec
# Load punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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
    """Removes punctuation and stop words"""
    # Remove punctuation
    letters_only = re.sub("[^a-zA-Z]", # non-letters
                        " ", # avoid fusing letters
                        job_description)

    # Split into list of lower case words
    lower_case = letters_only.lower().split()
    
    # Remove stop words
    words = [w for w in lower_case if w not in stop_words]
    
    return (" ".join(words))
    
stops = stopwords.words('english')
# Add lables to stop words
additional_stops = ['engineer', 'engineering', 'engineers', 'analyses', 'analysis',
    'analyst', 'analytic', 'analytical', 'analytics', 'analyze', 
    'analyzing', 'scientist', 'science', 'scientist', 'scientists']
for stop in additional_stops:
    stops.append(stop)
# Into set
stops = set(stops)

# Clean job descriptions
descriptions['desc'] = descriptions.desc.apply(
    lambda x: preprocess_words(x, stops)
)

# Bag of words without lemmatization
vectorizer = TfidfVectorizer(
    max_features=500,
    sublinear_tf=True, # scale counts: 1 + log(count)
    strip_accents='unicode' 
)
# Learn vocabulary and  transform into features 
features = vectorizer.fit_transform(
    descriptions.desc).toarray()

# Vocabulary non-lemmatized
vocab = vectorizer.get_feature_names()
# count word appearances
counts = np.sum(features, axis=0)
for word, n in zip(vocab, counts):
    print(n, '-', word)
print('-'*75)


# Lemmatize words
lemmer = WordNetLemmatizer()
descriptions["desc_lemma"] = descriptions.desc.apply(
    lambda x: ' '.join([lemmer.lemmatize(word, pos='a') for word in x.split()])
)

# Bag of words with lemmatization
vectorizer_lemma = TfidfVectorizer(
    max_features=500,
    sublinear_tf=True
)
# Learn vocabulary and  transform into features 
features_lemmatized = vectorizer_lemma.fit_transform(
    descriptions.desc_lemma).toarray()


# Vocabulary lemmatized
vocab_lemma = vectorizer_lemma.get_feature_names()
print(vocab_lemma)
counts = np.sum(features_lemmatized, axis=0)
for word, n in zip(vocab_lemma, counts):
    print(n, '-', word)

# Random indices for train and test sets
n_features = descriptions.shape[0] # number of features
test_size = round(n_features*0.15) # test size of 15%
test_indices = np.random.choice(
    range(n_features), replace=False, size=test_size
)
train_indices = [x for x in range(n_features) if x not in test_indices]

# Random train/test split
test_features = features_lemmatized[test_indices]
train_features = features_lemmatized[train_indices]
targets = descriptions.loc[:, ['data_science', 'data_engineering', 'data_analytics',
               'ml_engineering']]
train_targets = targets.iloc[train_indices, :]
test_targets = targets.iloc[test_indices, :]

# Combine features and targets
train_set = pd.concat(
    [pd.DataFrame(train_features), train_targets.reset_index(drop=True)],
    axis=1)
test_set = pd.concat(
    [pd.DataFrame(test_features), test_targets.reset_index(drop=True)],
    axis=1)

# Save data
train_set.to_csv('NLPData/train_set_bow.csv')
test_set.to_csv('NLPData/test_set_bow.csv')
pd.DataFrame(vocab_lemma).to_csv('NLPData/vocab_bow.csv')

# Word2Vec
# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------

def desc_to_wordlist(job_description, remove_stopwords=False):
    """Returns list of lower case words"""
    # Remove punctuation
    letters_only = re.sub("[^a-zA-Z]", # non-letters
                        " ", # avoid fusing letters
                        job_description)

    # Split into list of lower case words
    words = letters_only.lower().split()
    
    # Remove stop words if remove_stopwords is set to True
    if remove_stopwords:
        words = [w for w in lower_case if w not in stop_words]
    
    return words
    
    

def preprocess_sentences(job_desc, tokenizer, remove_stopwords=False):
    """
    Split job description into parsed sentences.
    Returns list of sentences, with each sentence 
    represented as a list of words (list of lists)
    """
    # Split descriptions into sentences
    raw_sentences = tokenizer.tokenize(job_desc.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0: # skip empty
            sentences = sentences.append(
                desc_to_wordlist(raw_sentence, remove_stopwords)
            )
    return sentences
    
sentences = []
