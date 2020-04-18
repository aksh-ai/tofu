import os
import sys
sys.path.append("..")
import numpy as np
from tofu.layers import Linear, ReLU, Dropout
from tofu.nn import Sequential, Model
from tensorflow.keras.datasets import mnist as MNIST
import time
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = MNIST.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

def ANN(X):
	model = Sequential()
	model.add(Linear(X.shape[1], 224))
	model.add(ReLU())
	model.add(Linear(224, 128))
	model.add(ReLU())
	model.add(Linear(128, 10))
	return model

model = ANN(X_train)

hist = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), learning_rate=0.1, verbose=1)

plt.figure(figsize=(8,2))

plt.subplot(1, 2, 1)
plt.plot(hist['loss'], label='Train Loss')
plt.plot(hist['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Metrics')
plt.legend(loc='best')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(hist['accuracy'], label='Train Accuracy')
plt.plot(hist['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Metrics')
plt.legend(loc='best')
plt.grid()
plt.show()