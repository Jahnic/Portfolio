import pandas as pd 
import numpy as np
# Logisic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Bag of words
test = pd.read_csv('NLPData/test_set_bow.csv').drop('Unnamed: 0', axis=1)
train = pd.read_csv('NLPData/train_set_bow.csv').drop('Unnamed: 0', axis=1)
vocab = pd.read_csv('NLPData/vocab_bow.csv').drop('Unnamed: 0', axis=1)
vocab = vocab.values.flatten()


def logistic_regression(C):
    """
    Returns mean scores, models, coefficients and predictions
    of multinomial logistic regression on each of the class labels
    """
    scores = []
    models = []
    coefficients = pd.DataFrame({'words': vocab})
    predictions = pd.DataFrame()
    class_labels = ['data_science', 'data_engineering', 'data_analytics',
               'ml_engineering']
    # Features
    train_features = train.drop(class_labels, axis=1)
    test_features = test.drop(class_labels, axis=1)
    
    # Multinomial logistic regression
    for target_name in class_labels:
        # Targets
        train_target = train[target_name]
        test_target = test[target_name]
        classifier = LogisticRegression(
            C=C, solver='sag', multi_class='multinomial'
        )

        # Crossvalidation
        cv_score = np.mean(cross_val_score(
            classifier,
            train_features,
            train_target,
            cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        
        # Fit model
        classifier.fit(train_features, train_target)
        
        # Feature labels
        pred = (target_name + "_pred")
        act = (target_name + "_actual")
        # Predict
        predictions[pred] = classifier.predict_proba(test_features)[:, 1]
        predictions[act] = test_target
        # Append model and coefficients
        models.append(classifier)
        coefficients[target_name] = classifier.coef_[0]
    return (np.mean(scores), models, coefficients, predictions)

# Hyper parameter tuning
# for hyper_param in range(1, 100, 5):
#     C = hyper_param/100
#     print('Total CV score for C={} is {}'.format(C, lr(C)[0]))
    
results = logistic_regression(C=1)
auc = results[0]
coefs = results[2]
preds = results[3]

def correct_prediction(row):
    """
    Returns true if the feature with the highest probability 
    correctly identifies the job role
    """
    # Extract predictions and corresponding labels
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

# Top 20 coefficiensts
coefs.sort_values(by='data_science', ascending=False).head(20)
coefs.sort_values(by='data_analytics', ascending=False).head(20)
coefs.sort_values(by='data_engineering', ascending=False).head(20)
coefs.sort_values(by='ml_engineering', ascending=False).head(20)