import numpy as np
import time
from tofu.losses import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits, mse, grad_mse

class Model:
	def __init__(self, layers=[], loss='crossentropy'):
		self.layers = layers

		try:
			if loss == 'crossentropy':
				self.loss = softmax_crossentropy_with_logits
				self.grad_loss = grad_softmax_crossentropy_with_logits

			elif loss == 'mse':
				self.loss = mse
				self.grad_loss = grad_mse

			else:
				print("Invalid loss function defined")
		
		except:
			print("Error")
	
	def add(self, layer):
		self.layers.append(layer)

	def __forward(self, X):
		activations = []
		inputs = X

		for layer in self.layers:
			activations.append(layer(inputs))
			inputs = activations[-1]
		
		assert len(activations) == len(self.layers)
		return activations

	def predict(self, X):
		logits = self.__forward(X)[-1]
		return logits.argmax(axis=-1)

	def __train(self, X , y, learning_rate):
		layer_activations = self.__forward(X)
		layer_inputs = [X]+layer_activations  
		logits = layer_activations[-1]
		
		loss = self.loss(logits, y)
		loss_grad = self.grad_loss(logits, y)
		
		for layer_index in range(len(self.layers))[::-1]:
			layer = self.layers[layer_index]
			
			loss_grad = layer.backward(inputs=layer_inputs[layer_index], grad_out=loss_grad, learning_rate=learning_rate) 
			
		return np.mean(loss)

	def __create_batches(self, inputs, targets, batch_size, shuffle=False):
		assert len(inputs) == len(targets)
		
		if shuffle:
			indices = np.random.permutation(len(inputs))
		
		for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
			if shuffle:
				excerpt = indices[start_idx:start_idx + batch_size]
			
			else:
				excerpt = slice(start_idx, start_idx + batch_size)
			
			yield inputs[excerpt], targets[excerpt]

	def validate(self, X, y):
		logits = self.__forward(X)[-1]
		loss = self.loss(logits, y)
		return np.mean(loss)

	def fit(self, X_train, y_train, epochs=30, batch_size=32, validation_data=(), learning_rate=0.1, verbose=5):        
		train_acc = []
		train_loss = []
		val_acc = [] 
		val_loss = []

		X_test, y_test = validation_data[0], validation_data[1]

		tot_start = time.time()

		for epoch in range(epochs):
			start_time = time.time()

			for x_batch, y_batch in self.__create_batches(X_train, y_train, batch_size=32, shuffle=True):
				t_loss = self.__train(x_batch, y_batch, learning_rate)
			
			train_acc.append(np.mean(self.predict(X_train) == y_train))
			val_acc.append(np.mean(self.predict(X_test) == y_test))
			train_loss.append(t_loss)
			val_loss.append(self.validate(X_test, y_test))

			end_time = time.time() - start_time

			if epoch==0 or epoch==(epochs-1) or (epoch % verbose == 0):
				print(f"Epoch: {epoch+1}")
				print(f"accuracy: {train_acc[-1]:.4f}  loss: {train_loss[-1]:.4f}  val_accuracy: {val_acc[-1]:.4f}  val_loss: {val_loss[-1]:.4f}  time: {(end_time/60):.2f} mins")

		print(f"\nTotal Training Duration: {(time.time() - tot_start) / 60:.2f} mins")
		return {'accuracy': train_acc, 'val_accuracy': val_acc, 'loss': train_loss, 'val_loss': val_loss}

class Sequential(Model):
	def __init__(self, layers=[], loss='crossentropy'):
		super().__init__()