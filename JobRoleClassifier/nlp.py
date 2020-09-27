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

)

# Vocabulary non-lemmatized
vocab = vectorizer.get_feature_names()
# count word appearances
counts = np.sum(features, axis=0)
for word, n in zip(vocab, counts):
    print(n, '-', word)
print('-'*75)

# Bag of words with lemmatization
vectorizer_lemma = TfidfVectorizer(
    max_features=500,
    sublinear_tf=True
)
# Learn vocabulary and  transform into features 
features_lemmatized = vectorizer_lemma.fit_transform(
    descriptions.desc_lemma).toarray()

# Lemmatize words
lemmer = WordNetLemmatizer()
descriptions["desc_lemma"] = descriptions.desc.apply(
    lambda x: ' '.join([lemmer.lemmatize(word, pos='a') for word in x.split()])
)
# Vocabulary lemmatized
vocab_lemma = vectorizer_lemma.get_feature_names()
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

# Train/test split
test_features = features_lemmatized[test_indices]
train_features = features_lemmatized[train_indices]

target = descriptions["data_science"]
train_target = target[train_indices]
test_target = target[test_indices]

def lr(C):
    # Multinomial logistic regression
    scores = []
    models = []
    coefficients = pd.DataFrame({'words': vocab_lemma})
    predictions = pd.DataFrame(index=test_indices)
    
    for target_name in class_names:
        target = descriptions[target_name]
        train_target = target[train_indices]
        test_target = target[test_indices]
        classifier = LogisticRegression(C=C, solver='sag', multi_class='auto') # regularization

        # Crossvalidation
        cv_score = np.mean(cross_val_score(
            classifier,
            train_features,
            train_target,
            cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        
        # Fit model
        classifier.fit(train_features, train_target)
        
        pred = (target_name + "_pred")
        act = (target_name + "_actual")
        # Predict
        predictions[pred] = classifier.predict_proba(test_features)[:, 1]
        predictions[act] = test_target
        models.append(classifier)
        coefficients[target_name] = classifier.coef_[0]
    return (np.mean(scores), models, coefficients, predictions)

# Hyper parameter tuning
# for hyper_param in range(1, 100, 5):
#     C = hyper_param/100
#     print('Total CV score for C={} is {}'.format(C, lr(C)[0]))
    
results = lr(1)
coefs = results[2]
preds = results[3]

def correct_prediction(row):
    pred = list(row[[0, 2, 4, 6]])
    act = list(row[[1, 3, 5, 7]])

    # Index of value 1
    act_index = act.index(1)
    # Indices of max prediction probability and true value correspond
    if pred[act_index] == max(pred):
        return True
    else:
        return False

outcomes = preds.apply(correct_prediction, axis=1)
accuracy = outcomes.sum()/outcomes.shape[0]
round(accuracy*100)