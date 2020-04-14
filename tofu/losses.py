import numpy as np

def mae(y_true, y_pred):
	num_samples = y_true.size
	loss = np.sum(np.abs(y_true - y_pred)) / num_samples
	return loss

def mse(y_true, y_pred):
	num_samples = y_true.size
	loss = np.sum((y_true - y_pred)**2) / num_samples
	return loss	

def rmse(y_true, y_pred):
	num_samples = y_true.size
	loss = np.sum((y_true - y_pred)**2) / num_samples
	return np.sqrt(loss)

def binary_crossentropy(y_true, y_pred):
	loss = 0.0
	
	for i in range(len(y_true)):
		loss += y_true[i] * np.log(1e-15 + y_pred[i])
	
	mean_loss = 1.0 / len(y_true) * loss
	
	return -mean_loss

def categorical_crossentropy(y_true, y_pred):
	loss = 0.0
	
	for i in range(len(y_true)):
		for j in range(len(y_true[i])):
			loss += y_true[i][j] * np.log(1e-15 + y_pred[i][j])
	
	mean_loss = 1.0 / len(y_true) * loss
	
	return -mean_loss	