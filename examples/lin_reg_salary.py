import sys
import os
sys.path.append("..")
sys.path.append("..")
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tofu.modules.linear_model import LinearRegression
from tofu.preprocessing import StandardScaler, MinMaxScaler, train_test_split

df = pd.read_csv('dataset/salary_data.csv')

print(df.head())

std_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()

X = std_scaler.fit_transform(df['experience'].values.reshape(-1,1))
y = min_max_scaler.fit_transform(df['salary'].values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, test_size=0.3, random_seed=101)

print(len(X_train), len(X_test), len(y_train), len(y_test),'\n')
print("STD: ", std_scaler.sigma)
print("MEAN: ", std_scaler.mu)

plt.scatter(X_train, y_train)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

model = LinearRegression()

model.fit(X_train, y_train, learning_rate=0.1, epochs=100)

plt.plot(model.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Metrics')
plt.show()

line = model.predict(X_train)

test_pred = model.predict(X_test)

loss = (np.square(test_pred - y_test)).mean(axis=None)

print(f"\nTest Loss MSE: {loss:.4f}")

plt.scatter(X_train, y_train)
plt.plot(X_train, line, c='r')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Line of Best Fit')
plt.show()