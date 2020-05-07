import os
import sys
sys.path.append("..")
import numpy as np
from tofu.layers import Linear, LeakyReLU, BatchNormalization
from tofu.nn import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

def ANN(X):
	model = Sequential()
	model.add(Linear(X.shape[1], 224))
	model.add(BatchNormalization())
	# model.add(ReLU())
	model.add(LeakyReLU(alpha=0.3))
	model.add(Linear(224, 128))
	model.add(BatchNormalization())
	# model.add(ReLU())
	model.add(LeakyReLU(alpha=0.3))
	model.add(Linear(128, 10))
	return model

model = ANN(X_train)

model.load_weights('models/MNIST_tofu.h5')

pred = model.predict(X_test[0])

print(f"Predicted Label: {pred} | Actual Label: {y_test[0]}")