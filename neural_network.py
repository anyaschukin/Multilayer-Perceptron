import pandas as pd
import numpy as np
import tools as tools
import matplotlib.pyplot as plt


# def relu_prime(z):
#     if z > 0:
#         return 1
#     return 0

# def cost(yHat, y):
#     return 0.5 * (yHat - y)**2

# def cost_prime(yHat, y):
#     return yHat - y

## binary cross-entropy loss??
# def entropy_loss(self,y, yhat):
#     nsample = len(y)
#     loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
#     return loss

NUM_NEURONS = 4

# The ReLufunction performs a threshold operation to each input element 
# where values less than zero are set to zero.
def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_prime(Z):
    return 1 * (Z > 0)
    # return 1 if z > 0 else 0

# leaky_ReLU
def leaky_ReLU(z, alpha = 0.01):
	return np.where(z >= 0, 1, z * alpha)
    # return max(alpha * z, z)

def leaky_ReLU_prime(z, alpha = 0.01):
    return np.where(z >= 0, 1, alpha)
	# return 1 if z > 0 else alpha

# sigmoid
def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

# sigmoid derivative
def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

    # return np.exp(z) / np.sum(np.exp(z)) 

	# e = np.exp(float(z))
	# return (e/np.sum(e))

def softmax_prime(z):
  return softmax(z) * (1-softmax(z))

# derivative for f is f(1-f) for respective cost functions
# THIS IS CORRECT FOR sigmoid and softmax
# def activation_derivative(activation):
    # return activation * (1.0 - activation)

# def activation_derivative(z, f):
# 	if fn == SIGMOID:
# 		f = sigmoid
# 	elif fn == SOFTMAX:
# 		f = softmax
# 	return f(z)*(1-f(z))



def compute_loss(y_hat, y):
    return ((y_hat - y)**2).sum()


class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], NUM_NEURONS) 
        self.weights2   = np.random.rand(NUM_NEURONS,1)
        self.weights3   = np.random.rand(NUM_NEURONS,1)
        self.weights4   = np.random.rand(NUM_NEURONS,1)
        # self.bias1       = np.zeros((self.input.shape[0], NUM_NEURONS))
        self.bias1       = np.zeros((NUM_NEURONS, 1))
        self.bias2       = np.zeros((y.shape[0], 1))
        self.bias3       = np.zeros((y.shape[0], 1))
        self.bias4       = np.zeros((y.shape[0], 1))
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self, activation = softmax):
        
        self.layer1 = activation(np.dot(self.input, self.weights1) + self.bias1) 
        self.layer2 = activation(np.dot(self.layer1.T, self.weights2) + self.bias2)
        self.layer3 = activation(np.dot(self.layer2.T, self.weights3) + self.bias3)
        self.output = activation(np.dot(self.layer3.T, self.weights4) + self.bias4) # layer = theta(weight_l * a_l-1 + b_l)

    ## application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    def backprop(self, activation_derivative = softmax_prime, learning_rate = 0.1):

        # output layer
        d_Z4 = self.output - y
        d_weights4 = np.dot(self.layer3.T, d_Z4)
        d_bias4 = np.sum(d_Z4, axis = 1, keepdims=True)

        # hidden layers
        d_Z3 = np.dot(self.weights3.T, d_Z4) * activation_derivative(self.layer3)
        d_weights3 = np.dot(self.layer2.T, d_Z3) # (layer-1) * output error
        d_bias3 = np.sum(d_Z3, axis = 1, keepdims=True)
        
        d_Z2 = np.dot(self.weights2.T, d_Z3) * activation_derivative(self.layer2)
        d_weights2 = np.dot(self.layer1.T, d_Z2) # (layer-1) * output error
        d_bias2 = np.sum(d_Z2, axis = 1, keepdims=True)

        d_Z1 = np.dot(self.weights2.T, d_Z2) * activation_derivative(self.layer1)
        d_weights1 = np.dot(self.input.T, d_Z1) # (layer-1) * output error
        d_bias1 = np.sum(d_Z2, axis = 1, keepdims=True)
       
        ## update the weights with the derivative (slope) of the loss function
        self.weights1 -= learning_rate * d_weights1
        self.weights2 -= learning_rate * d_weights2
        self.weights3 -= learning_rate * d_weights3
        self.weights4 -= learning_rate * d_weights4

        self.bias1 -= learning_rate * d_bias1
        self.bias2 -= learning_rate * d_bias2
        self.bias3 -= learning_rate * d_bias3
        self.bias4 -= learning_rate * d_bias4


X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(X,y)

loss_values = []

for i in range(1500):
    nn.feedforward()
    nn.backprop()
    loss = compute_loss(nn.output, y)
    loss_values.append(loss)

print(nn.output)
print(f" final loss : {loss}")

plt.plot(loss_values)
plt.show()





# # softmax activation layer : compute values for each sets of scores in x
# # not sure this works


# a_layer = theta(weight_l * a_l-1 + b_l)


