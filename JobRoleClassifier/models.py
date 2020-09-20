import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
#Cross validation
from sklearn.model_selection import train_test_split
# Linear models
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression, Lasso
# Random forest
from sklearn.ensemble import RandomForestRegressor
# Model tuning and scores
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error



df = pd.read_csv("data/data_clean.csv")
df.columns

# choose relevant columns
df_model = df[['avg_salary', 'title', 'locational_salary_quantiles', 'Rating', 'Size', 'Type of ownership', 'Industry',
            'Sector', 'Revenue', 'seniority', 'desc_len', 'age', 'python', 'R', 'spark',
            'aws', 'excel', 'sql']]
# get dummy data
df_dum = pd.get_dummies(df_model)

# train  test split
X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# multiple linear regression
X_sm = X = sm.add_constant(X) # intercept
model = sm.OLS(y, X_sm)
model.fit().summary()
# Sklern LR
lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error')) 
# lasso regression
lm_l = Lasso(alpha=0.19)
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error')) 

alpha = []
error = []

for i in range(1, 100):
    alpha.append(i/100)
    lm_l = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(
        lm_l, X_test, y_test, scoring='neg_mean_absolute_error')) 
)
plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]
 
# random forest
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train))

# tune forest model
parameters = {'n_estimators': range(10, 300, 10), 'criterion': ('mse', 'mae'),
              'max_features': ('auto','sqrt', 'log2')}

gs = GridSearchCV(rf, parameters, scoring='neg_mean_absolute_error')
gs.fit(X_train, y_train)

gs.best_score_
gs.best_estimator_

# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_lml)
mean_absolute_error(y_test, tpred_rf)

mean_absolute_error(y_test, (0.9*tpred_rf + 0.1*tpred_lm))
