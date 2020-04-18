import numpy as np

class LinearRegression:
	def __init__(self):
		# attributes
		self.total_cost = 0
		self.w = 0
		self.b = 0
		self.losses = []

	def __slope(self, w, x, b):
		'''
		performs slope calculation y = m * x + b
		x = Input Features
		w = Weights
		b = Bias
		'''
		out = np.matmul(x.transpose(), w) + b
		return out

	def __cost(self, x, y, w, b):
		'''
		computes Mean Sqared Error for given X & y per iteration
		'''
		self.total_cost = 0
		num_samples = len(x)

		for i in range(0, num_samples):
			predicted = self.__slope(x[i], w, b)[0]
			self.total_cost += (y[i] - predicted) ** 2

		self.total_cost = self.total_cost / float(num_samples)

		return self.total_cost	

	def __compute_grad(self, x, y, w, b, learning_rate):
		'''
		performs Gradient Descent to optimize Weights & Bias paramters per iteration
		'''
		dw, db = 0, 0
		num_samples = len(x)

		for i in range(0, num_samples):
			dw += - (2 *  x[i] * (y[i] - self.__slope(x=x[i], w=w, b=b)[0])) / num_samples
			db += - (2 * (y[i] - self.__slope(x=x[i], w=w, b=b)[0])) / num_samples

		w = w - learning_rate * dw.reshape(x.shape[1], 1)
		b = b - learning_rate * db

		return w, b		

	def fit(self, x, y, learning_rate=0.001, epochs=1000):
		'''
		Training function
		'''
		# Initialize/Sample weights and biases from a random normal distribution
		# Xavier Initialization
		# Square Root(6 / (input features + output features))
		w = np.random.normal(loc=0.0, scale=np.sqrt(6/(x.shape[1] + 1)), size=(x.shape[1], 1))
		b = np.random.normal(loc=0.0, scale=np.sqrt(6/(x.shape[1] + 1)), size=1)[0]

		# Train the model for given epochs
		for _ in range(0, epochs):
			# calculate loss
			self.losses.append(self.__cost(x=x, y=y, w=w, b=b))
			# perform Gradient Descent to optimize Weights & Biases
			w, b = self.__compute_grad(x=x, y=y, w=w, b=b, learning_rate=learning_rate)

		# Update class's weights & biases with optimized weights & biases
		self.w = w
		self.b = b

		# store losses by reshaping it as a vector of values
		self.losses = np.array(self.losses).reshape(-1, 1)

	def predict(self, x):
		'''
		returns predicted values when input with new data points
		'''
		return np.matmul(x, self.w) + self.b 

class LogisticRegression:
	def __init__(self):
		# attributes
		self.total_cost = 0
		self.w = 0
		self.b = 0
		self.losses = []

	def __slope(self, w, x, b):
		'''
		performs slope calculation y = m * x + b
		x = Input Features
		w = Weights
		b = Bias
		'''
		out = np.matmul(x.transpose(), w) + b
		return out

	def __sigmoid(self, z):
		'''
		performs Sigmoid/Logistic fucntion over input value
		'''
		return 1.0 / (1 + np.exp(-z))

	def __binary_crossentropy(self, y, h):
		'''
		returns binary cross-entropy value
		'''
		return (-y * np.log(h) - (1 - y) * np.log(1 - h))

	def __cost(self, x, y, w, b):
		'''
		Cost/Loss function - Computes Binary Cross-Entropy
		'''
		self.total_cost = 0
		num_samples = len(x)

		for i in range(0, num_samples):
			predicted = self.__slope(x[i], w, b)[0]
			self.total_cost += self.__binary_crossentropy(y=y[i], h=self.__sigmoid(predicted))

		self.total_cost = self.total_cost / float(num_samples)

		return self.total_cost	

	def __compute_grad(self, x, y, w, b, learning_rate):
		'''
		performs Gradient Descent to optimize Weights & Bias paramters per iteration
		'''
		dw, db = 0, 0
		num_samples = len(x)

		for i in range(0, num_samples):
			dw += np.dot(x[i].transpose(), (self.__sigmoid(self.__slope(x=x[i], w=w, b=b)[0]) - y[i])) / num_samples
			db += (self.__sigmoid(self.__slope(x=x[i], w=w, b=b)[0]) - y[i]) / num_samples

		w -= learning_rate * dw.reshape(x.shape[1], 1)
		b -= learning_rate * db

		return w, b		

	def fit(self, x, y, learning_rate=0.001, epochs=1000):
		'''
		Training function
		'''
		# Initialize/Sample weights and biases from a random normal distribution
		# Xavier Initialization
		# Square Root(6 / (input features + output features))
		w = np.random.normal(loc=0.0, scale=np.sqrt(6/(x.shape[1] + 1)), size=(x.shape[1], 1))
		b = np.random.normal(loc=0.0, scale=np.sqrt(6/(x.shape[1] + 1)), size=1)[0]

		# Train the model for given epochs
		for _ in range(0, epochs):
			# calculate loss
			self.losses.append(self.__cost(x=x, y=y, w=w, b=b))	
			# perform Gradient Descent to optimize Weights & Biases
			w, b = self.__compute_grad(x=x, y=y, w=w, b=b, learning_rate=learning_rate)

		# Update class's weights & biases with optimized weights & biases
		self.w = w
		self.b = b

		# store losses by reshaping it as a vector of values
		self.losses = np.array(self.losses).reshape(-1, 1)

	def predict(self, x):
		'''
		returns predicted class when input with new data points
		'''
		return self.__sigmoid(np.matmul(x, self.w) + self.b).reshape(x.shape[0], 1) >= 0.5