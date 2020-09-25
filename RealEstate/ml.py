import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# linear regression
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# Deep learning 
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader 
import imageio
# Saves model as file
import pickle


# Load Data
df = pd.read_csv("data/model_data.csv").iloc[:, 4: ]
# df = pd.read_csv("/home/jahnic/Git/Portfolio/RealEstate/data/condos.csv").iloc[:, 4: ]
# Copy coordinates in separate table and drop from df

# Define features and target
y = df.price.values
X = df.drop('price', axis=1).astype('float')

# Split data 
X_train, X_test, y_train, y_test  = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Statsmodels for further feature selection 
X_sm = sm.add_constant(X) # intercept
model = sm.OLS(y, X_sm)
model.fit().summary()



# Train LR model
LR = LinearRegression()
LR.fit(X_train, y_train)

# Predict test values
predicted = LR.predict(X_test)

# R^2
r2 = LR.score(X_test, y_test)
print("R-Squared:", r2)

# Predicted vs. actual
y_pred = pd.Series(predicted.flatten())
y_actual = pd.Series(y_test)
# Merge into data frame
results = pd.concat([y_actual, y_pred], axis=1, ignore_index=True)
results.columns = ['Actual', 'Predicted']
# Save results
results.to_csv("data/linear_regression_results.csv")
filename = 'linear_regression_model.sav'
pickle.dump(LR, open(filename, 'wb'))
# MSE
mse = mean_squared_error(y_true=y_actual, y_pred=y_pred)
print("Squareroot of MSE:", np.sqrt(mse))

# # Coefficient weights *use only with scaled features*
# feature_list = X.columns
# importance = LR.coef_
# # Assign coefficient weights to corresponding features
# feature_importance = {}
# for i, feature in enumerate(feature_list):
#     feature_importance[feature] = importance[i]
# # Print feature importance    
# for feature, score in feature_importance.items():
# 	print('Feature: {} Score: {:.5f}'.format(feature, score))
 
# # plot feature importance
# plt.bar(feature_list, importance)
# plt.xticks(rotation=90)
# plt.show()

# Pytorch prediction
EPOCHS = 200
BATCH_SIZE = X.shape[1] # number of features

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

y_train = torch.from_numpy(np.array(y_train).astype(np.float32))
torch.FloatTensor(y_train)

# train data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train), torch.FloatTensor(y_train))


# test data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = testData(torch.FloatTensor(X_test))



train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

CUDA_LAUNCH_BLOCKING=1
net = torch.nn.Sequential(
        torch.nn.Linear(BATCH_SIZE, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1)
    )
net.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

net.train()
for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = net(X_batch)
        
        loss = loss_func(y_pred, y_batch.unsqueeze(1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}')

# Save nn model 
# filename = 'nn_model.sav'
# pickle.dump(net, open(filename, 'wb'))    
    
y_pred_list = []
net.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = net(X_batch)
#       print(y_test_pred)
        y_pred_list.append(y_test_pred.cpu().numpy())
        
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

mse = mean_squared_error(y_test, y_pred_list)
np.sqrt(mse)
BATCH_SIZE

