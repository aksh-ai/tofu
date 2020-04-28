import numpy as np

class Linear:
	def __init__(self, shape_1, shape_2, name=None):
		self.name = name
		self.__shape_1 = shape_1
		self.__shape_2 = shape_2
		self.__lim = np.sqrt(6/(0.5 + shape_1 + shape_2))
		
		self.weights = np.random.uniform(low= -self.__lim, high= self.__lim, size=(self.__shape_1, self.__shape_2))
		self.bias = np.zeros(shape=self.__shape_2)

	def __call__(self, inputs):
		return np.matmul(inputs, self.weights) + self.bias

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
		self.__indices = None

	def __call__(self, inputs):
		dropped = inputs.flatten()
		num_drop = self.prob * inputs.size
		self.__indices = np.random.choice(len(dropped), size=int(num_drop))
		
		for i in self.__indices:
			dropped[i] = 0 
			
		dropped = dropped * (1/(1 - self.prob))

		return dropped.reshape(inputs.shape)

	def backward(self, inputs, grad_out, learning_rate):
		for i in self.__indices:
			grad_out[i] = 0 
			
		grad_out = grad_out * (1 / (1 - self.prob))
		
		return grad_out

class BatchNormalization:
	def __init__(self, momentum=0.99, epsilon=1e-8, axis=0, training=True, name=None):
		self.name = name

		self.momentum = momentum
		
		self.gamma = 1.0
		self.beta = 0.0
		self.epsilon = epsilon
		
		self.axis = axis
		
		self.mu = None
		self.var = None

		self.training = True

	def __call__(self, inputs):
		if self.training == False:
			self.X_norm = (inputs - self.mu) / np.sqrt(self.var + self.epsilon)
			return self.gamma * self.X_norm + self.beta
		
		self.mu = np.mean(inputs, axis=self.axis)
		self.var = np.var(inputs, axis=self.axis)

		self.X_norm = (inputs - self.mu) / (np.sqrt(self.var + self.epsilon) ** 0.5)
		
		out = self.gamma * self.X_norm + self.beta

		self.mu = self.mu * self.momentum + self.mu * (1 - self.momentum)
		self.var = self.var * self.momentum + self.var * (1 - self.momentum)

		return out

	def backward(self, inputs, grad_out, learning_rate):
		N = inputs.shape[0]

		X_mu = inputs - self.mu
		std_inv = 1. / np.sqrt(self.var + self.epsilon)

		dX_norm = grad_out * self.gamma
		
		d_var = np.sum(dX_norm * X_mu, axis=self.axis) * -.5 * std_inv**3
		d_mu = np.sum(dX_norm * -std_inv, axis=self.axis) + d_var * np.mean(-2. * X_mu, axis=self.axis)

		dX = (dX_norm * std_inv) + (d_var * 2 * X_mu / N) + (d_mu / N)
		
		dgamma = np.sum(grad_out * self.X_norm, axis=0)
		dbeta = np.sum(grad_out, axis=0)

		self.gamma -= learning_rate * dgamma
		self.beta -= learning_rate * dbeta

		return dX 

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
		out = np.ones_like(inputs)
		out[inputs < 0] = self.alpha
		return grad_out * out

class ELU:
	def __init__(self, alpha=0.3):
		self.alpha = alpha

	def __call__(self, inputs):
		return np.where(inputs<=0, self.alpha * (np.exp(inputs) - 1), inputs)

	def backward(self, inputs, grad_out, learning_rate):
		out = np.where(inputs<=0, self.alpha * np.exp(inputs), 0)
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