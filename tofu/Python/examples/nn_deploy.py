import os
import sys
sys.path.append("..")
import numpy as np
from layers import Linear, LeakyReLU, BatchNormalization
from nn import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

def ANN(X):
	model = Sequential()
	model.add(Linear(X.shape[1], 256))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.25))
	model.add(Linear(256, 512))
	model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.25))
	model.add(Linear(512, 10))
	return model

model = ANN(X_train)

model.load_weights('models/MNIST.h5')

pred = model.predict(X_test[0])

print(f"Predicted Label: {pred} | Actual Label: {y_test[0]}")

pred = model.predict(X_test[10])

print(f"Predicted Label: {pred} | Actual Label: {y_test[10]}")

pred = model.predict(X_test[50])

print(f"Predicted Label: {pred} | Actual Label: {y_test[50]}")

pred = model.predict(X_test[132])

print(f"Predicted Label: {pred} | Actual Label: {y_test[132]}")