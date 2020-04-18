import numpy as np

def mae(y_true, y_pred):
	num_samples = y_true.size
	loss = np.sum(np.abs(y_true - y_pred)) / num_samples
	return loss

def mse(y_true, y_pred):
	num_samples = y_true.size
	loss = np.sum((y_true - y_pred)**2) / num_samples
	return loss	

def grad_mse(y_true, y_pred):
	num_samples = y_true.size
	loss = (2 * np.sum(np.abs(y_true - y_pred))) / num_samples
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

def softmax_crossentropy_with_logits(logits, true_logits):
    nlogits = logits[np.arange(len(logits)), true_logits]
    
    entropy = - nlogits + np.log(np.sum(np.exp(logits), axis=-1))
    
    return entropy

def grad_softmax_crossentropy_with_logits(logits, true_logits):
    nlogits = np.zeros_like(logits)
    nlogits[np.arange(len(logits)), true_logits] = 1
    
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    
    return (- nlogits + softmax) / logits.shape[0]