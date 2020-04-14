import os
import sys
sys.path.append("..")
import numpy as np
from tofu.layers import Dropout
from tofu.activations import relu, leaky_relu, elu

drop = Dropout(prob=0.5)

inputs = np.random.uniform(low=0.1, high=5.0, size=(3, 3))

print(inputs,'\n\n')

dropped = drop(inputs)

print(dropped)

# act_drop = relu(dropped)
# act_drop = leaky_relu(dropped, alpha=0.2)
act_drop = elu(dropped, alpha=0.2)

print(act_drop)