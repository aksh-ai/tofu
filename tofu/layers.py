import numpy as np

class Linear:
	def __init__(self, shape_1, shape_2, name=None):
		self.name = name
		self.shape_1 = shape_1
		self.shape_2 = shape_2
		
		self.weights = np.random.normal(loc=0.0, scale=np.sqrt(6/(shape_1 + shape_2)), size=(shape_1, shape_2))
		self.bias = np.random.normal(loc=0.0, scale=np.sqrt(6/(shape_1 + shape_2)), size=shape_2)
		
		self.params = [self.weights, self.bias]

	def __call__(self, inputs):
		return np.dot(inputs, self.weights) + self.bias

	def backward(self, inputs, grad_out, learning_rate):
		grad_in = np.dot(grad_out, self.weights.transpose())

		dw = np.dot(inputs.transpose(), grad_out).reshape(self.weights.shape)
		db = (grad_out.mean(axis=0) * inputs.shape[0]).reshape(self.bias.shape)

		self.weights -= learning_rate * dw
		self.bias -= learning_rate * db

		return grad_in

class Dropout:
	def __init__(self, prob=0.5, name=None):
		self.name = name
		self.prob = prob

	def __call__(self, inputs):
		dropped = inputs.flatten()
		num_drop = self.prob * inputs.size
		indices = np.random.choice(len(dropped), size=int(num_drop))
		
		for i in indices:
			dropped[i] = np.zeros(1, dtype=np.float32)[0]

		return dropped.reshape(inputs.shape)

	def backward(self, inputs, grad_out, learning_rate):
		return grad_out	

class BatchNormalization:
	def __init__(self, momentum=0.99, epsilon=1e-5, axis=-1, training=True, name=None):
		self.name = name

		self.momentum = momentum
		self.gamma = np.random.normal(loc=0.0, scale=0.02, size=1)[0]
		self.beta = np.random.normal(loc=0.0, scale=0.02, size=1)[0]
		self.epsilon = epsilon
		
		self.axis = axis
		
		self.mu = None
		self.sigma = None

		self.training = True

		self.params = np.array([self.gamma, self.beta])

	def __call__(self, X):
		if self.training==True:
			self.mu = X.mean(axis=self.axis)
			self.sigma = (X - self.mu)**2 / X.size

		else:
			self.mu = self.mu * self.momentum + self.mu * (1 - self.momentum)
			self.sigma = self.sigma * self.momentum + self.sigma * (1 - self.momentum)

		X_hat = (X - self.mu) / np.sqrt(self.sigma + self.epsilon)

		y =	self.gamma * X_hat + self.beta

		return y

	def backward(self, inputs, grad_out, learning_rate):
		grad_in = self.gamma * inputs + self.beta

		d_gamma = (grad_out * self.gamma)
		d_beta = (grad_out.mean(axis=0) * inputs.shape[0])

		self.gamma -= learning_rate * d_gamma
		self.beta -= learning_rate * d_beta

		return grad_in

class ReLU:
	def __init__(self):
		pass

	def __call__(self, inputs):
		return np.maximum(0, inputs)

	def backward(self, inputs, grad_out, learning_rate):
		out = inputs > 0
		return grad_out * out

class LeakyReLU:
	def __init__(self, alpha=0.3):
		self.alpha = alpha

	def __call__(self, inputs):
		return np.maximum(inputs, inputs * self.alpha)

	def backward(self, inputs, grad_out, learning_rate):
		out = 1 if inputs.any() > 0 else self.alpha
		return grad_out * out

class ELU:
	def __init__(self, alpha=0.3):
		self.alpha = alpha

	def __call__(self, inputs):
		return np.maximum(inputs, self.alpha*(np.exp(inputs) - 1))

	def backward(self, inputs, grad_out, learning_rate):
		out = 1 if inputs.any() > 0 else self.alpha * np.exp(inputs) #+ inf
		return grad_out * out

class TanH:
	def __init__(self):
		pass

	def __call__(self, inputs):
		return (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))

	def backward(self, inputs, grad_out, learning_rate):
		out = 1 - np.power(self.__call__(inputs), 2)
		return grad_out * out

class Sigmoid:
	def __init__(self):
		pass

	def __call__(self, inputs):
		return 1.0 / (1 + np.exp(-inputs))

	def backward(self, inputs, grad_out, learning_rate):
		out = self.__call__(inputs) * (1 - self.__call__(inputs))
		return grad_out * out