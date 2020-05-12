import time
import pickle
import numpy as np
from losses import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits, mse, grad_mse

class Model:
	def __init__(self, layers=[], loss='crossentropy'):
		self.layers = layers

		try:
			if loss == 'crossentropy':
				self.__loss = softmax_crossentropy_with_logits
				self.__grad_loss = grad_softmax_crossentropy_with_logits

			elif loss == 'mse':
				self.__loss = mse
				self.__grad_loss = grad_mse

			else:
				print("Invalid loss function defined")
		
		except:
			print("Error")
	
	def add(self, layer):
		self.layers.append(layer)

	def __forward(self, X):
		try:
			activations = []
			inputs = X

			for layer in self.layers:
				activations.append(layer(inputs))
				inputs = activations[-1]
			
			return activations

		except Exception as e:
			print(f"Error in forwarding through layers\n\n{e}\n")	

	def predict(self, X):
		if self.__loss=='mse':
			logits = self.__forward(X)[-1]
			return logits[0]

		logits = self.__forward(X)[-1]
		return logits.argmax(axis=-1)

	def __train(self, X , y, learning_rate):
		layer_activations = self.__forward(X)
		layer_inputs = [X] + layer_activations  
		logits = layer_activations[-1]
		
		loss = self.__loss(logits, y)
		loss_grad = self.__grad_loss(logits, y)
		
		for layer_index in range(len(self.layers))[::-1]:
			layer = self.layers[layer_index]
			
			loss_grad = layer.backward(inputs=layer_inputs[layer_index], grad_out=loss_grad, learning_rate=learning_rate) 
			
		return np.mean(loss)

	def __create_batches(self, inputs, targets, batch_size, shuffle=False):
		try:
			if shuffle:
				indices = np.random.permutation(len(inputs))
			
			for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
				if shuffle:
					excerpt = indices[start_idx:start_idx + batch_size]
				
				else:
					excerpt = slice(start_idx, start_idx + batch_size)
				
				yield inputs[excerpt], targets[excerpt]

		except Exception as e:
			print(f"Error in creating batches\n\n{e}\n")		

	def validate(self, X, y):
		logits = self.__forward(X)[-1]
		loss = self.__loss(logits, y)
		return np.mean(loss)

	def fit(self, X_train, y_train, epochs=30, batch_size=32, validation_data=(), learning_rate=0.1, verbose=5):        
		print(f"{int(len(X_train)/batch_size)} batch of samples per epoch | {len(X_train)} number of samples in total")

		train_acc = []
		train_loss = []
		val_acc = [] 
		val_loss = []
		t_loss = None

		X_test, y_test = validation_data[0], validation_data[1]

		tot_start = time.time()

		for epoch in range(epochs):
			start_time = time.time()

			for x_batch, y_batch in self.__create_batches(X_train, y_train, batch_size=batch_size, shuffle=True):
				t_loss = self.__train(x_batch, y_batch, learning_rate)
			
			train_acc.append(np.mean(self.predict(X_train) == y_train))
			val_acc.append(np.mean(self.predict(X_test) == y_test))
			train_loss.append(t_loss)
			val_loss.append(self.validate(X_test, y_test))

			end_time = time.time() - start_time

			if epoch==0 or epoch==(epochs-1) or (epoch % verbose == 0):
				print(f"Epoch {epoch+1}")
				print(f"accuracy: {train_acc[-1]:.4f} - loss: {train_loss[-1]:.4f} - val_accuracy: {val_acc[-1]:.4f} - val_loss: {val_loss[-1]:.4f} - time: {(end_time/60):.2f} mins")

		print(f"\nTotal Training Duration: {(time.time() - tot_start) / 60:.2f} mins")
		return {'accuracy': train_acc, 'val_accuracy': val_acc, 'loss': train_loss, 'val_loss': val_loss}

	def save_weights(self, path):
		self.layer_names = []
		self.weights_dict = {}
		count = 0

		for layer in self.layers:
			if layer.trainable:
				if layer.name in self.layer_names:
					count += 1
				else:
					self.layer_names.append(layer.name)

				self.weights_dict['{}_{}'.format(layer.name, count)] = layer.parameters

		with open(path, 'wb') as f:
			pickle.dump(self.weights_dict, f)

	def load_weights(self, path):
		self.weights_dict = pickle.load(open(path, 'rb'))
		names = list(self.weights_dict.keys())
		cnt = 0

		for layer in self.layers:
			if layer.trainable:
				layer.parameters = self.weights_dict[names[cnt]]
				cnt += 1

				try:
					layer.weights = layer.parameters['weights']
					layer.biases = layer.parameters['biases']

				except Exception as e:
					layer.gamma = layer.parameters['gamma']
					layer.beta = layer.parameters['beta']

class Sequential(Model):
	def __init__(self, layers=[], loss='crossentropy'):
		super().__init__(layers, loss)