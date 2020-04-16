import numpy as np

class LinearRegression:
	def __init__(self):
		self.total_cost = 0
		self.w = 0
		self.b = 0
		self.losses = []

	def slope(self, w, x, b):
		out = np.matmul(x.transpose(), w) + b
		return out

	def cost(self, x, y, w, b):
		self.total_cost = 0
		num_samples = len(x)

		for i in range(0, num_samples):
			predicted = self.slope(x[i], w, b)
			self.total_cost += (y[i] - predicted) ** 2

		self.total_cost = self.total_cost / float(num_samples)

		return self.total_cost	

	def fit(self, x, y, learning_rate=0.001, epochs=1000):
		w = np.random.normal(loc=0.0, scale=0.0, size=(x.shape[1], 1))
		b = np.random.normal(loc=0.0, scale=0.0, size=1)[0]

		for i in range(0, epochs):
			self.losses.append(self.cost(x=x, y=y, w=w, b=b))	
			w, b = self.compute_grad(x=x, y=y, w=w, b=b, learning_rate=learning_rate)

		self.w = w
		self.b = b

	def compute_grad(self, x, y, w, b, learning_rate):
		dw, db = 0, 0
		num_samples = len(x)

		for i in range(0, num_samples):
			dw += - (2/num_samples) *  np.sum(np.dot(x[i].reshape(-1, 1), (y[i] - self.slope(x=x[i], w=w, b=b))).reshape(-1, 1))
			db += - (2/num_samples) * (y[i] - self.slope(x=x[i], w=w, b=b))

		w = w - learning_rate * dw
		b = b - learning_rate * db

		return w, b		

	def predict(self, x):
		w = self.w
		b = self.b
		return self.slope(x=x.transpose(), w=w, b=b)

class LogisticRegression:
	def __init__(self):
		self.total_cost = 0
		self.w = 0
		self.b = 0
		self.losses = []

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x)) 	

	def slope(self, w, x, b):
		try:
			out = np.dot(x.transpose(), w) + b
			
		except:
			out = np.dot(w, x) + b

		try:
			return np.sum(out, axis=1)

		except:
			return np.sum(out.reshape(-1, 1))	

	def cost(self, x, y, w, b):
		self.total_cost = 0
		num_samples = len(x)

		for i in range(0, num_samples):
			predicted = self.slope(x[i], w, b)
			self.total_cost += (y[i] - predicted) ** 2

		self.total_cost = self.total_cost / float(num_samples)

		return self.total_cost	

	def fit(self, x, y, learning_rate=0.001, epochs=1000):
		w = np.random.normal(loc=0.0, scale=0.0, size=(x.shape[1], 1))
		b = np.random.normal(loc=0.0, scale=0.0, size=1)[0]

		for i in range(0, epochs):
			self.losses.append(self.cost(x=x, y=y, w=w, b=b))	
			w, b = self.compute_grad(x=x, y=y, w=w, b=b, learning_rate=learning_rate)

		self.w = w
		self.b = b

	def compute_grad(self, x, y, w, b, learning_rate):
		dw, db = 0, 0
		num_samples = len(x)

		for i in range(0, num_samples):
			dw += - (2/num_samples) *  np.sum(np.dot(x[i].reshape(-1, 1), (y[i] - self.sigmoid(self.slope(x=x[i], w=w, b=b)))).reshape(-1, 1))
			db += - (2/num_samples) * (y[i] - self.sigmoid(self.slope(x=x[i], w=w, b=b)))

		w = w - learning_rate * dw
		b = b - learning_rate * db

		return w, b		

	def predict(self, x):
		w = self.w
		b = self.b

		predictions = self.sigmoid(self.slope(x, w, b)).reshape(x.shape[0], 1)

		for i in range(0, len(predictions)):
			if predictions[i]<0.5:
				predictions[i] = 0
			else:
				predictions[i] = 1	
		
		return predictions