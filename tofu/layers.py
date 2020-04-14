import numpy as np

class Linear:
	def __init__(self, shape_1, shape_2, name=None):
		self.name = name
		self.shape_1 = shape_1
		self.shape_2 = shape_2
		self.weights = np.random.uniform(low=0.1, high=1.0, size=(shape_1, shape_2))
		self.bias = np.random.uniform(low=0.1, high=1.0, size=1)[0]
		self.params = [self.weights, self.bias]

	def __call__(self, inputs):
		out = np.dot(self.weights.transpose(), inputs) + self.bias
		return np.sum(out, axis=1)

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

		return dropped.reshape(inputs.shape[0], inputs.shape[1])