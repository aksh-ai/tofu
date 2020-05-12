import numpy as np

def relu(x):
	return np.maximum(0, x)

def leaky_relu(x, alpha=0.3):
	return np.maximum(x, alpha*x)

def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

def elu(z, alpha=0.3):
	return np.maximum(z, alpha*(np.exp(z) - 1))

def linear(x, m=1):	
	return x * m

def tanh(z):
	return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)