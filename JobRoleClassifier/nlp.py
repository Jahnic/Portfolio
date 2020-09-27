import pandas as pd 
import numpy as np
# Preprocessing
import re
# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Logisic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
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
class_names = ['data_science', 'data_engineering', 'data_analytics',
               'ml_engineering']
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
    
stops = set(stopwords.words('english'))
# Clean job descriptions
descriptions['desc'] = descriptions.desc.apply(
    lambda x: preprocess_words(x, stops)
)

# Bag of words without lemmatization
vectorizer = TfidfVectorizer(
    max_features=200,
    sublinear_tf=True, # scale counts: 1 + log(count)
    strip_accents='unicode' 
)
# Learn vocabulary and  transform into features 
features = vectorizer.fit_transform(
    descriptions.desc).toarray()

# Bag of words with lemmatization
vectorizer_lemma = TfidfVectorizer(
    max_features=200,
    sublinear_tf=True
)
# Lemmatize words
lemmer = WordNetLemmatizer()
descriptions["desc_lemma"] = descriptions.desc.apply(
    lambda x: ' '.join([lemmer.lemmatize(word, pos='a') for word in x.split()])
)
# Learn vocabulary and  transform into features 
features_lemmatized = vectorizer_lemma.fit_transform(
    descriptions.desc_lemma).toarray()

# Vocabulary non-lemmatized
vocab = vectorizer.get_feature_names()
# count word appearances
counts = np.sum(features, axis=0)
for word, n in zip(vocab, counts):
    print(n, '-', word)
print('-'*75)
# Vocabulary lemmatized
vocab_lemma = vectorizer_lemma.get_feature_names()
counts = np.sum(features_lemmatized, axis=0)
for word, n in zip(vocab_lemma, counts):
    print(n, '-', word)

# Random indices for tain and test sets
n_features = descriptions.shape[0] # number of features
test_size = round(n_features*0.15) # test size of 15%
test_indices = np.random.choice(
    range(n_features), replace=False, size=test_size
)
train_indices = [x for x in range(n_features) if x not in test_indices]

# Train/test split
test_features = features_lemmatized[test_indices]
train_features = features_lemmatized[train_indices]


def lr(C):
    # Multinomial logistic regression
    scores = []
    predictions = pd.DataFrame({'id': test_indices})
    
    for target_name in class_names:
        target = descriptions[target_name]
        train_target = target[train_indices]
        classifier = LogisticRegression(C=C, solver='sag') # regularization

        # Crossvalidation
        cv_score = np.mean(cross_val_score(
            classifier,
            train_features,
            train_target,
            cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        # print('CV score for class {} is {}'.format(target_name, cv_score))

        classifier.fit(train_features, train_target)
        predictions[target_name] = classifier.predict_proba(test_features)[:, 1]
        return np.mean(scores)

# Hyper parameter tuning
for hyper_param in range(1, 100, 5):
    C = hyper_param/100
    print('Total CV score for C={} is {}'.format(C, lr(C)))
# print feature importance
for word, coefficient in zip()
