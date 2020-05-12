import os
import sys
sys.path.append("..")
import numpy as np
from layers import Linear, Dropout, BatchNormalization
from activations import tanh

class ANN:
	def __init__(self):
		# ANN Layers
		self.fc1 = Linear(3, 3)
		self.dropout = Dropout(prob=0.4)
		self.batch_norm = BatchNormalization()
		self.fc2 = Linear(3, 1)

	def forward(self, X):
		# performs single forward pass through the network through all layers as a chain
		out = self.fc1(X)
		# print(out,'\n')
		out = self.dropout(out)
		# print(out,'\n')
		out = self.batch_norm(out)
		# print(out,'\n')
		out = self.fc2(out)
		# print(out,'\n')
		out = tanh(out)
		# print(out,'\n')
		return out

model = ANN()

inputs = np.random.uniform(low=0.1, high=50.0, size=(3, 3))

# print(inputs,'\n')

out = model.forward(inputs)

print(f"Ouput value: {out[0, 0]:.5f}")