import random
import numpy as np

class StandardScaler:
	def __init__(self):
		self.mu =  None
		self.sigma = None
		
	def fit(self, X):
		X = np.array(X)
		self.sigma = X.std()
		self.mu = X.mean()

	def transform(self, X):
		X = np.array(X)
		X = (X - self.mu) / self.sigma
		return X

	def fit_transform(self, X):
		X = np.array(X)
		self.sigma = X.std()
		self.mu = X.mean()
		X = (X - self.mu) / self.sigma
		return X

	def inverse_transform(self, X):
		X = np.array(X)
		return (X * self.sigma)	+ self.mu

class MinMaxScaler:
	def __init__(self, feature_range=(0, 1)):
		self.min =  feature_range[0]
		self.max = feature_range[1]
		self.scale = None

	def fit(self, X):
		X = np.array(X)
		self.scale = (self.max - self.min) / (X.max(axis=0) - X.min(axis=0))

	def transform(self, X):
		X = np.array(X)
		X_scaled = self.scale * X + min - X.min(axis=0) * self.scale
		return X_scaled		
	
	def fit_transform(self, X):
		X = np.array(X)
		self.scale = (self.max - self.min) / (X.max(axis=0) - X.min(axis=0))
		X_scaled = self.scale * X + self.min - X.min(axis=0) * self.scale
		return X_scaled

def train_test_split(X, y, train_size=None, test_size=None, random_seed=None):
	if len(X)!=len(y):
		print("Number of samples and labels don't match")
		return

	if train_size:
		test_size = round((1.0 - train_size), 1)
		
	test_size = int(len(y) * test_size)

	X_test = X[0:test_size]
	X_train = X[test_size:len(X)]
	y_test = y[0:test_size]
	y_train = y[test_size:len(y)]

	X_train, y_train = shuffle(X_train, y_train, random_seed)
	X_test, y_test = shuffle(X_test, y_test, random_seed)

	return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def shuffle(a=None, b=None, random_seed=None):
	if random_seed:
		random.seed(random_seed)

	if a is not None and b is not None:
		shuffler = list(zip(a, b))
		random.shuffle(shuffler)
		a, b = zip(*shuffler)
		return np.array(a), np.array(b)

	elif a is not None:
		random.shuffle(a)
		return np.array(a)

	elif b is not None:
		random.shuffle(b)
		return np.array(b)

	else:
		print('Array(s) are empty')
		return None