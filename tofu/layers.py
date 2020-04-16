import numpy as np

class Linear:
	def __init__(self, shape_1, shape_2, name=None):
		self.name = name
		self.shape_1 = shape_1
		self.shape_2 = shape_2
		
		self.weights = np.random.normal(loc=0.0, scale=0.2, size=(shape_1, shape_2))
		self.bias = np.random.normal(loc=0.0, scale=0.2, size=1)[0]
		
		self.params = np.array([self.weights, self.bias])

	def __call__(self, inputs):
		out = np.dot(self.weights.transpose(), inputs) + self.bias
		try:
			return np.sum(out, axis=1)

		except:
			return np.sum(out.reshape(-1, 1))	

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

		return dropped.reshape(inputs.size)

class BatchNormalization:
	def __init__(self, momentum=0.99, epsilon=1e-5, axis=-1, training=True, name=None):
		self.name = name

		self.momentum = momentum
		self.gamma = np.random.normal(loc=0.0, scale=0.0, size=1)[0]
		self.beta = np.random.normal(loc=0.0, scale=0.0, size=1)[0]
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