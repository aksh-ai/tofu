import os
import sys
sys.path.append("..")
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tofu.modules.linear_model import LogisticRegression
from tofu.preprocessing import StandardScaler, train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('dataset/binary.csv')

print(df.head())

std_scaler = StandardScaler()

X = std_scaler.fit_transform(df['score'].values.reshape(-1,1))
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, train_size=0.7, random_seed=101)

plt.scatter(X_train, y_train)
plt.xlabel('Score')
plt.ylabel('Target')
plt.show()

print(len(X_train), len(X_test), len(y_train), len(y_test),'\n')
print("STD: ", std_scaler.sigma)
print("MEAN: ", std_scaler.mu)

model = LogisticRegression()

model.fit(X_train, y_train, learning_rate=0.01, epochs=30)

plt.plot(model.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Metrics')
plt.show()

train_pred = model.predict(X_train)

test_pred = model.predict(X_test)

# print(train_pred,'\n')
# print(test_pred)

print(f'\nConfusion Matrix for Training set:\n {confusion_matrix(train_pred, y_train)} \n')
print(f'Confusion Matrix for Test set:\n {confusion_matrix(test_pred, y_test)} \n')