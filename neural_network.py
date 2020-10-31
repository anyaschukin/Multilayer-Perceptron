import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import json

# from tools import parse_args
from preprocess import split, split_x_y
from activations import *
from validation_metrics import *

LAYER1_NEURONS = 16
LAYER2_NEURONS = 16
LAYER3_NEURONS = 16
LAYER4_NEURONS = 2

class NeuralNetwork:
	def __init__(self, num_features, batch_size, epochs, learning_rate = 0.01):
		self.batch_size	 	= batch_size
		self.epochs			= epochs
		self.learning_rate  = learning_rate
		self.input          = None
		self.weights1       = np.random.rand(LAYER1_NEURONS, num_features) * np.sqrt(2/num_features) ## * 0.01
		self.weights2       = np.random.rand(LAYER2_NEURONS, LAYER1_NEURONS) * np.sqrt(2/LAYER1_NEURONS) ## * 0.01
		self.weights3       = np.random.rand(LAYER3_NEURONS, LAYER2_NEURONS) * np.sqrt(2/LAYER2_NEURONS) ## * 0.01
		self.weights4       = np.random.rand(2, LAYER3_NEURONS) * np.sqrt(2/LAYER3_NEURONS) ## * 0.01  # if multiple classification, it should (NUM_NEURONS, output_size(# of classes))

		self.bias1          = np.zeros((LAYER1_NEURONS, 1))
		self.bias2          = np.zeros((LAYER2_NEURONS, 1))
		self.bias3          = np.zeros((LAYER3_NEURONS, 1))
		self.bias4          = np.zeros((2, 1)) # (num_of_classes, 1)
	   
		self.y              = None
		self.output         = np.zeros((2, batch_size))

	def load_model(self, model):
		self.weights1       = np.array(model['weights1'])
		self.weights2       = np.array(model['weights2'])
		self.weights3       = np.array(model['weights3'])
		self.weights4       = np.array(model['weights4'])

		self.bias1          = np.array(model['bias1'])
		self.bias2          = np.array(model['bias2'])
		self.bias3          = np.array(model['bias3'])
		self.bias4          = np.array(model['bias4'])

	def feedforward(self, activation = softmax, activation_hidden = sigmoid):
		
		self.Z1 = np.dot(self.weights1, self.input.T) + self.bias1
		self.layer1 = activation_hidden(self.Z1) 

		self.Z2 = np.dot(self.weights2, self.layer1) + self.bias2
		self.layer2 = activation_hidden(self.Z2)
		
		self.Z3 = np.dot(self.weights3, self.layer2) + self.bias3
		self.layer3 = activation_hidden(self.Z3)

		self.Z4 = np.dot(self.weights4, self.layer3) + self.bias4 # layer = theta(weight_l * a_l-1 + b_l)
		self.output = activation(self.Z4) # layer = theta(weight_l * a_l-1 + b_l)


	## application of the chain rule to find derivative of the loss function, with respect to weights and bias
	def backprop(self, d_activation = softmax_prime, d_activation_hidden = sigmoid_prime):

		m = self.batch_size # num examples or batch size

		d_A4 = compute_loss_prime(self.output, self.y)

		# output layer
		d_Z4 = d_A4 * d_activation(self.Z4)
		d_weights4 = np.dot(d_Z4, self.layer3.T)
		d_bias4 = np.sum(d_Z4, axis = 1, keepdims=True)
		d_A3 = np.dot(self.weights4.T, d_Z4)

		# hidden layers
		d_Z3 = d_A3 * d_activation_hidden(self.Z3)
		d_weights3 = np.dot(d_Z3, self.layer2.T)  # (layer-1) * output error
		d_bias3 = np.sum(d_Z3, axis = 1, keepdims=True)
		d_A2 = np.dot(self.weights3.T, d_Z3)
 
		d_Z2 = d_A2 * d_activation_hidden(self.Z2)
		d_weights2 = np.dot(d_Z2, self.layer1.T) # (layer-1) * output error
		d_bias2 = np.sum(d_Z2, axis = 1, keepdims=True) 
		d_A1 = np.dot(self.weights2.T, d_Z2)

		d_Z1 = d_A1 * d_activation_hidden(self.Z1)
		d_weights1 = np.dot(d_Z1, self.input) # (layer-1) * output error
		d_bias1 = np.sum(d_Z2, axis = 1, keepdims=True)
	   
		## update the weights with the derivative (slope) of the loss function (SGD)
		self.weights1 -= self.learning_rate * (d_weights1 / m) 
		self.weights2 -= self.learning_rate * (d_weights2 / m)
		self.weights3 -= self.learning_rate * (d_weights3 / m)
		self.weights4 -= self.learning_rate * (d_weights4 / m)

		self.bias1 -= self.learning_rate * (d_bias1 / m)
		self.bias2 -= self.learning_rate * (d_bias2 / m)
		self.bias3 -= self.learning_rate * (d_bias3 / m)
		self.bias4 -= self.learning_rate * (d_bias4 / m)


	def train(self, data, args, train_set, test_set, num_examples):
		# train_set, test_set = split(data)
		train_losses, test_losses = [], []
		for epoch in range(self.epochs):
			shuffle(train_set)
			for i in range(0, num_examples, self.batch_size):
				self.input, self.y = split_x_y(train_set[i:i+self.batch_size]) ######### batch_size or self.batch_size?
				self.feedforward()
				self.backprop()
				
				train_loss = compute_loss(self.output[:, 0], self.y[:, 0])
				train_losses.append(train_loss)

			test_loss = compute_loss(self.output[:, 0], self.y[:, 0])
			test_losses.append(test_loss)

			# print validation metrics 'epoch - train loss - test loss'
			# print("epoch {}/{}: train loss = {}, test loss = {}".format(epoch, self.epochs, round(train_loss, 4), round(test_loss, 4)))

		if args.evaluation:
			y_pred = probability_to_class(self.output.T)
			get_validation_metrics(y_pred[:, 0], self.y.T[:, 0])

		if not args.mini_batch:
			plot_learning(train_losses, test_losses)

		if args.save_model:
			# save network params
			W1, W2, W3, W4 = self.weights1.tolist(), self.weights2.tolist(), self.weights3.tolist(), self.weights4.tolist()
			B1, B2, B3, B4 = self.bias1.tolist(), self.bias2.tolist(), self.bias3.tolist(), self.bias4.tolist()
			model = dict(weights1=W1, weights2=W2, weights3=W3, weights4=W4, bias1=B1, bias2=B2, bias3=B3, bias4=B4)
			with open("neural_network.json", "w") as f:
				json.dump(model, f, separators=(',', ':'), indent=4)


	def predict(self, test_set):
		num_examples = test_set.shape[0]

		shuffle(test_set)
		self.input, self.y = split_x_y(test_set)
		self.output = np.zeros((2, num_examples))

		# replicate feedforward for testing
		self.feedforward()
		test_loss = compute_loss(self.output[:, 0], self.y[:, 0])
		print("\n" + colors.LGREEN + "Final loss on validation set = {}".format(test_loss) + colors.ENDC + "\n")
