import os
import sys
sys.path.append("..")
import numpy as np
from tofu.layers import Linear, LeakyReLU, BatchNormalization
from tofu.nn import Sequential
from tofu.preprocessing import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

X_train, y_train = shuffle(X_train, y_train, random_seed=101)
X_test, y_test = shuffle(X_test, y_test, random_seed=101)

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

hist = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test), learning_rate=0.1, verbose=1)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

ax1.plot(hist['loss'], label='Train Loss')
ax1.plot(hist['val_loss'], label='Validation Loss')
ax1.set(xlabel='Epochs', ylabel='Loss')
ax1.set_title('Loss Metrics')
ax1.legend(loc='best')
ax1.grid()

ax2.plot(hist['accuracy'], label='Train Accuracy')
ax2.plot(hist['val_accuracy'], label='Validation Accuarcy')
ax2.set(xlabel='Epochs', ylabel='Accuracy')
ax2.set_title('Accuracy Metrics')
ax2.legend(loc='best')
ax2.grid()

fig.tight_layout()
# plt.savefig('nn_test.png')
plt.show()

model.save_weights('models/MNIST.h5')