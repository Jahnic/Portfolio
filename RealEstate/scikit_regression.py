import pandas as pd 
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 35
pd.options.display.float_format = "{:.6f}".format
# Saves model as file
import pickle
# Outlier detection
from sklearn.ensemble import IsolationForest
#Stats
from scipy.stats import skew
from scipy.stats.stats import pearsonr

# Load data
path = "data/complete_data.csv"
complete_data = pd.read_csv(path).iloc[:, 3: ]

# Transform is_devided into boolean feature
complete_data['is_devided'] = complete_data.is_devided.astype('bool')


drop_cols = ['less_than_$50,000_(%)', 'between_$50,000_and_$80,000_(%)', 
                'between_$80,000_and_$100,000_(%)', 'between_$100,000_and_$150,000_(%)',
                'more_than_$150,000_(%)', '1-person_households_(%)', 
                '2-person_households_(%)', '3-person_households_(%)', 
                '4-person_households_(%)', '5-person_or_more_households_(%)', 
                'couples_without_children_at_home_(%)', 'couples_with_children_at_home_(%)',
                'single-parent_families_(%)', 'owners_(%)', 'renters_(%)',
                'before_1960_(%)', 'between_1961_and_1980_(%)',
                'between_1981_and_1990_(%)', 'between_1991_and_2000_(%)',
                'between_2001_and_2010_(%)', 'between_2011_and_2016_(%)',
                'single-family_homes_(%)', 'semi-detached_or_row_houses_(%)',
                'buildings_with_less_than_5_floors_(%)',
                'buildings_with_5_or_more_floors_(%)', 'mobile_homes_(%)',
                'university_(%)', 'college_(%)', 'secondary_(high)_school_(%)',
                'apprentice_or_trade_school_diploma_(%)', 'no_diploma_(%)',
                'non-immigrant_population_(%)', 'immigrant_population_(%)',
                'french_(%)', 'english_(%)', 'others_languages_(%)',
                'new_area_from_price', 'new_area_from_rooms', 'basement_bedroom',
                'n_parking', 'population_2016_',
                'rooms','lat', 'long', 'restaurants', 'shopping', 'vibrant', 'cycling_friendly',
                'car_friendly', 'historic', 'quiet', 'elementary_schools',
                'high_schools', 'parks', 'nightlife', 'groceries', 'daycares',
                'pedestrian_friendly', 'cafes', 'transit_friendly', 'greenery']
data = complete_data.drop(drop_cols, axis=1)
cluster_data = pd.read_csv('data/cluster_data.csv', index_col='Unnamed: 0')
data = pd.concat([data, cluster_data.iloc[:, : 8]], axis=1)
# Remove symbols violating tf scope naming conventions
valid_column_names = [col.replace('_(%)', '').replace('$', 'CAD').
                      replace(',', '.').
                      replace('(', '').replace(')', '') for col in data.columns]
data.columns = valid_column_names
# Remove price outliers
data = data[data.price <= 5000000]
# Random index shuffling for train/test split
df = data.copy().sample(frac=1, random_state=0)
# Prepare train and test data
train_size = round(0.8*df.shape[0])
train_indices = df[: train_size].index
test_indices = df[train_size : ].index
# Ensure proper lengths
(len(train_indices) + len(test_indices)) == data.shape[0]

# ----------------------> Preprocessing <---------------------- #

# Visualize log transformed vs raw price
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":data["price"], "log(price + 1)":np.log1p(data["price"])})
prices.hist()


#log transform skewed numeric features:
numeric_feats = data.dtypes[(data.dtypes != "bool") & 
                            (data.dtypes != "object")].index

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
# positive skew
positive_skewed_feats = skewed_feats[skewed_feats > 0.75]
log_transform = ['price', 'total_area', 'mr_distance',
                 'year_built', 'walk_score', 'population_density_',
                 'population_variation_between_2011_2016_']

# Transform with natural log + 1
fixed_skew_data = data.copy()
fixed_skew_data[log_transform] = np.log1p(fixed_skew_data[log_transform])
# sqrt transform PC_1
# Add +4 to make all values positive
fixed_skew_data['PC_neighborhood_1'] = np.sqrt(4 + data[['PC_neighborhood_1']])
# Show result
fixed_skew_data[numeric_feats].hist()

# Categorical data
cat_data = ['bedrooms', 'bathrooms', 'powder_rooms']
room_dummies = pd.get_dummies(data[cat_data].astype('str'))
fixed_skew_data.drop(cat_data, axis=1, inplace=True)
fixed_skew_data = pd.concat([fixed_skew_data, room_dummies], axis=1)

# Sklearn metrices
X_train = fixed_skew_data.loc[train_indices, 'price':]
X_test = fixed_skew_data.loc[test_indices, 'price':]
y_train = X_train.pop('price')
y_test = X_test.pop('price')

# -------------------------> Ridge and Lasso <------------------------ #

# Import models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, 
                                    X_train, 
                                    y_train, 
                                    scoring="neg_mean_squared_error", 
                                    cv = 5))
    return(rmse)

# Ridge
alphas = [0.01, 0.1, 0.3, 1, 3, 4.1, 5, 10, 15]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

# Visualize results
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")

# Lasso
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train)
print('Ridge min:', cv_ridge.min())
print('Lasso min:', rmse_cv(model_lasso).min())

# Lasso coefficients
coef = pd.Series(model_lasso.coef_, index = X_train.columns)
print("Lasso picked " 
      + str(sum(coef != 0)) 
      + " variables and eliminated the other " 
      +  str(sum(coef == 0)) 
      + " variables")

# Most important predictors
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

# Residuals
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
preds = pd.DataFrame({"preds":np.expm1(model_lasso.predict(X_test)),
                      "actual":np.expm1(y_test)})
preds["residuals"] = preds["actual"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
preds.residuals.std()


# -----------------------> xgBOOST <----------------------- #

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Hyperparameter optimization
params = {'max_depth': [2, 3, 4], \
          "eta": [0.005, 0.001, 0.01, 0.02], \
          "reg_alpha": [ 0.01, 0.025, 0.05, 0.1],
          "n_estimators": [1000]} #the params are tuned using xgb.cv
                         
model_xgb = xgb.XGBRegressor()

# Hyperparam tuning
clf = GridSearchCV(model_xgb, params, n_jobs=5,
                   scoring='neg_mean_squared_error', verbose=2, refit=True)
clf.fit(X_train, y_train)
clf.best_params_

# Build model
model_xgb = xgb.XGBRegressor(n_estimators=1000, 
                             max_depth=4,
                             eta=0.02,
                             reg_alpha=0.01) #the params were tuned using GridSearchCV
model_xgb.fit(X_train, y_train)

# Boost and lasso
xgb_preds = np.expm1(model_xgb.predict(X_test))
# lasso_preds = np.expm1(model_lasso.predict(X_test))
results = pd.DataFrame({'actual': np.expm1(y_test), 'pred': xgb_preds})
plt.scatter(results[results.actual < 1200000].actual, 
            results[results.actual < 1200000].pred)

# Error of boost residuals with condows below $1.2M
boost_error = (results[results.actual < 1200000].pred - 
               results[results.actual < 1200000].actual).std()
print("Boost test error:", boost_error)
# lasso_error = (np.expm1(y_test) - lasso_preds).std()
xgb.plot_importance(model_xgb)

# Save and load boosted model
model_name = 'web-app/boost_model.dat'
pickle.dump(model_xgb, open(model_name, 'wb'))
loaded_model = pickle.load(open(model_name, 'rb'))

# # Tree plot
# xgb.plot_tree(loaded_model)

# -----------------------> predictions on entire data set <------------#

# Categorical room data
prediction_data = data.iloc[:, 1 :].copy()
room_dummies = pd.get_dummies(prediction_data[cat_data].astype('str'))
prediction_data.drop(cat_data, axis=1, inplace=True)
prediction_data = pd.concat([prediction_data, room_dummies], axis=1)

# Transform skewed data
prediction_data['PC_neighborhood_1'] = np.sqrt(4 + prediction_data[['PC_neighborhood_1']])
prediction_data[log_transform] = np.log1p(prediction_data[log_transform])
boolean_feats = ['river_proximity', 'has_pool', 'has_garage', 'is_devided']
prediction_data[boolean_feats] = prediction_data[boolean_feats].astype('int')

features = prediction_data.copy()
target = features.pop('price')

# Prediction and prediction difference 
loaded_model = pickle.load(open(model_name, 'rb'))
predictions = loaded_model.predict(features)
prediction_table = pd.DataFrame({
                            'actual': np.expm1(target), 
                            'predicted': np.expm1(predictions)})
prediction_table['prediction_difference'] = (prediction_table.predicted 
                                  - prediction_table.actual)

# Concat prediction differences to entire data
complete_data_new = complete_data[complete_data.price <= 5000000].copy()
complete_data_new = pd.concat([complete_data_new, prediction_table], axis=1)
complete_data_new.drop(['actual', 'new_area_from_price', 'new_area_from_rooms'], 
                   axis=1, inplace=True)

print("Prediction difference variation:", complete_data_new.prediction_difference.std())

# Save data
complete_data_new.to_csv('web-app/data/data_with_prediction_differences.csv')

# ------------------------> keras <--------------------------- #

from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train = StandardScaler().fit_transform(X_train)

# Train/validation split 
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, random_state = 0)
X_tr.shape

model = Sequential()
# model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
model.add(Dense(1, input_dim = X_train.shape[1], activity_regularizer=l1(0.0005)))
model.compile(loss = "mse", optimizer = "adam")

hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val), epochs=20)
(np.expm1(model.predict(X_val)[:,0]) - np.expm1(y_val)).std()

