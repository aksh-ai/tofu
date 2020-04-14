import os
import sys
sys.path.append("..")
import numpy as np
from tofu.layers import Linear

inputs = np.random.uniform(low=0.1, high=5.0, size=(3, 3))

linear1 = Linear(3, 3)

print(f"Weights: \n{linear1.params[0]} \n\nBias: {linear1.params[1]:.5f}\n")

out = linear1(inputs)

print(f"Output:\n{out}\n")		

print(f"Argmax:\n{np.argmax(out[0])}\n")