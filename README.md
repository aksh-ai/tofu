# Tofu

Tofu is a framework for **Machine Learning** and **Deep Learning** supporting Python, C, and C++ programming languages. It is still in its very
early development stages.

## Description

Tofu contains the following features/functionalities as of now and will be upgraded in the future.

### modules 

* #### linear_model

    * Contains Linear Regression & Logistic Regression classes for multiple features.
 
### preprocessing

The preprocessing module mimics Scikit-Learn's preprocessing classes and functions.

* #### StandardScaler

    * Feature scaling using Standard Distribution.

* #### MinMaxScaler

    * Feature scaling by feature normalization using minimum and maximum values.

* #### train_test_split

    * Split given arrays of data and labels into training set and testing set.

* #### shuffle
    * Shuffles input arrays (single or a pair) and returns them.

### layers

Neural Network layers for Multi-layered perceptron models.

* #### Linear

    * Linear class implementing a single Fully-Connected / Dense layer.

* #### Dropout
	
	* Regularization layer that randomly drops neurons / sets neurons to zero randomly based on given probability. 

* #### Batch Normalization

    * Regularization layer that normalizes input batch of neurons.

* #### Activation Layers
    * ReLU
    * LeakyReLU
    * ELU
    * Sigmoid
    * TanH

### activations

A set of common activation functions are available in this script.

* Linear
* ReLU
* Leaky ReLU
* ELU
* Sigmoid
* TanH
* Softmax

### losses

A set of common loss functions are available in this script.

* Mean Squared Error (MSE) & Derivative
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Binary Cross-Entropy
* Categorical Cross-Entropy
* Softmax Cross-Entropy with logits & Derivative

## Requirements

Python / C / C++ development environment

Numpy is the only dependency for Python

`pip install numpy`

## Examples

I've added some example scripts and dataset under examples folder. Clone this repository and run the scripts from within the examples folder. Install the requirements using:

`pip install -r requirements.txt`

## About

tofu-v0.01-alpha