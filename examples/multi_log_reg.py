import os
import sys
sys.path.append("..")
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tofu.modules.linear_model import LogisticRegression
from tofu.preprocessing import StandardScaler, train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("dataset/advertising.csv")

print(df.head())

X = df[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']].values
y = df['Clicked on Ad'].values

std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, test_size=0.33, random_seed=101)

logm = LogisticRegression()

logm.fit(X_train, y_train, learning_rate=0.01, epochs=320)

plt.plot(logm.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Metrics')
plt.show()

predictions = logm.predict(X_test)

conmat = confusion_matrix(y_test, predictions)

print("Confusion matrix: \n", conmat)