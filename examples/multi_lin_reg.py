import sys
import os
sys.path.append("..")
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tofu.modules.linear_model import LinearRegression
from tofu.preprocessing import StandardScaler, MinMaxScaler, train_test_split
from tofu.losses import *

df = pd.read_csv("dataset/Ecommerce Customers")

X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']].values
y = df['Yearly Amount Spent'].values

print(df.head())

std_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

X = std_scaler.fit_transform(X)
y = min_max_scaler.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_seed = 101)

print(len(X_train), len(X_test), len(y_train), len(y_test),'\n')
print(X_train.shape, y_train.shape)

lm = LinearRegression()

lm.fit(X_train, y_train, learning_rate=0.01, epochs=100)

plt.plot(lm.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Metrics')
plt.show()

predictions = lm.predict(X_test)

print(predictions.shape)

print("\nMAE: ", mae(y_test, predictions))
print("MSE: ", mse(y_test, predictions))
print("RMSE: ", rmse(y_test, predictions))