import numpy as np

class LinearRegression:
	def __init__(self):
		self.total_cost = 0
		self.w_initial = 0
		self.b_initial = 0
		self.w = 0
		self.b = 0
		self.losses = []

	def slope(self, x, w, b):
		return (w * x + b)

	def cost(self, x, y, w, b):
		self.total_cost = 0
		num_samples = len(x)

		for i in range(0, num_samples):
			predicted = self.slope(x[i], w, b)
			self.total_cost += (y[i] - predicted) ** 2

		self.total_cost = self.total_cost/float(num_samples)

		return self.total_cost	

	def compute_grad(self, x, y, w, b, learning_rate):
		dw, db = 0, 0
		num_samples = len(x)

		for i in range(0, num_samples):
			dw += - (2/num_samples) *  x[i] * (y[i] - self.slope(x[i], w, b))
			db += - (2/num_samples) * (y[i] - self.slope(x[i], w, b))

		w = w - learning_rate * dw
		b = b - learning_rate * db

		return w, b	

	def fit(self, x, y, w=0, b=0, learning_rate=0.001, epochs=1000):
		for i in range(0, epochs):
			self.losses.append(self.cost(x, y, w, b))	
			w, b = self.compute_grad(x, y, w, b, learning_rate)

		self.w = w
		self.b = b

	def predict(self, x):
		w = self.w
		b = self.b
		return self.slope(x, w, b)	

class LogisticRegression:
	def __init__(self):
		self.total_cost = 0
		self.w_initial = 0
		self.b_initial = 0
		self.w = 0
		self.b = 0
		self.losses = []

	def slope(self, x, w, b):
		return (w * x + b)

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x)) 

	def cost(self, x, y, w, b):
		self.total_cost = 0
		num_samples = len(x)

		for i in range(0, num_samples):
			predicted = self.slope(x[i], w, b)
			self.total_cost += (y[i] - predicted) ** 2

		self.total_cost = self.total_cost / float(num_samples)

		return self.total_cost	

	def compute_grad(self, x, y, w, b, learning_rate):
		dw, db = 0, 0
		num_samples = len(x)

		for i in range(0, num_samples):
			dw += - (1/num_samples) *  x[i] * (y[i] - self.sigmoid(self.slope(x[i], w, b)))
			db += - (1/num_samples) * (y[i] - self.sigmoid(self.slope(x[i], w, b)))

		w = w - learning_rate * dw
		b = b - learning_rate * db

		return w, b	

	def fit(self, x, y, w=0, b=0, learning_rate=0.001, epochs=1000):
		for i in range(0, epochs):
			self.losses.append(self.cost(x, y, w, b))	
			w, b = self.compute_grad(x, y, w, b, learning_rate)

		self.w = w
		self.b = b

	def predict(self, x):
		w = self.w
		b = self.b
		
		predictions = self.sigmoid(self.slope(x, w, b))
		
		for i in range(0, len(predictions)):
			if predictions[i]<0.5:
				predictions[i] = 0
			else:
				predictions[i] = 1	
		
		return predictions