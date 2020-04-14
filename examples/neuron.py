import os
import sys
sys.path.append("..")
import numpy as np
from tofu.layers import Linear

inputs = np.random.uniform(low=0.1, high=10.0, size=(3, 3))

linear = Linear(3, 1)

print(f"Weights: \n{linear.params[0]} \n\nBias: {linear.params[1]:.5f}\n")

out = linear(inputs)

print(f"Output:\n{out}\n")		

print(f"Argmax:\n{np.argmax(out[0])}\n")