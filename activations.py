import pandas as pd
import numpy as np

def ReLU(z):
	return np.maximum(0, z)

def ReLU_prime(z):
	return 1 * (z > 0)

def leaky_ReLU(z, alpha = 0.1):
	return np.where(z >= 0, z, z * alpha)

def leaky_ReLU_prime(z, alpha = 0.1):
	return np.where(z >= 0, 1, alpha)

def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
	# Numerically Stable: (z - np.max(z) shifts the values of z so that the highest number is 0... [1, 3, 5] -> [-4, -2, 0]
	z_max = np.max(z, axis=0, keepdims=True)
	e = np.exp(z - z_max)
	return e / e.sum(axis=0, keepdims=True)

def softmax_prime(z):
	return softmax(z) * (1-softmax(z))

# binary cross-entropy loss
def compute_loss(yhat, y):
	m = len(y)
	loss = -1/m * (np.sum(np.dot(np.log(yhat).T, y) + np.dot((1 - y).T, np.log(1 - yhat))))
	return loss

def compute_loss_prime(yhat, y):
	d_loss = - (np.divide(y, yhat) - np.divide(1 - y, 1 - yhat))
	return d_loss