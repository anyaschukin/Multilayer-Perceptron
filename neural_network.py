import pandas as pd
import numpy as np
import tools as tools
import matplotlib.pyplot as plt
import preprocess as prep


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
        self.weights1   = np.random.rand(self.input.shape[1], NUM_NEURONS) * 0.01 # self.input.shape[1] is num_features
        self.weights2   = np.random.rand(NUM_NEURONS,1) * 0.01
        self.weights3   = np.random.rand(NUM_NEURONS,1) * 0.01
        self.weights4   = np.random.rand(NUM_NEURONS, 1) * 0.01
        # self.weights4   = np.random.rand(NUM_NEURONS, y.shape[0])
        # self.bias1       = np.zeros((self.input.shape[0], NUM_NEURONS))
        self.bias1       = np.random.rand(self.input.shape[0], 1)
        self.bias2       = np.random.rand(1, NUM_NEURONS)
        self.bias3       = np.random.rand(1, NUM_NEURONS)
        self.bias4       = np.random.rand(1, 1)
        # self.y          = y
        self.y          = np.zeros((y.shape[0], 1))
        self.output     = np.zeros((self.y.shape[0], 1))

    def feedforward(self, activation = softmax, activation_hidden = sigmoid):
        
        self.layer1 = activation_hidden(np.dot(self.input, self.weights1) + self.bias1) 
        # print(self.layer1.shape, self.weights1.shape)
        self.layer2 = activation_hidden(np.dot(self.layer1, self.weights2) + self.bias2)
        # print(self.layer2.shape, self.weights2.shape)
        self.layer3 = activation_hidden(np.dot(self.layer2, self.weights3) + self.bias3)
        print(self.layer3.shape)
        self.output = activation(np.dot(self.layer3, self.weights4) + self.bias4) # layer = theta(weight_l * a_l-1 + b_l)
        # print(self.output.shape, self.weights4.shape)

    ## application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    def backprop(self, activation_derivative = sigmoid_prime, learning_rate = 0.1):

        # output layer
        d_Z4 = self.output - self.y
        print(d_Z4.shape)
        d_weights4 = np.dot(self.layer3.T, d_Z4)
        d_bias4 = np.sum(d_Z4, axis = 1, keepdims=True)

        # hidden layers
        print(self.weights3.shape, d_Z4.shape)
        d_Z3 = np.dot(self.weights3, d_Z4.T) * activation_derivative(self.layer3)
        d_weights3 = np.dot(self.layer2, d_Z3.T) # (layer-1) * output error
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


def main():
    # X = np.array([[0,0,1],
    #               [0,1,1],
    #               [1,0,1],
    #               [1,1,1]])
    # y = np.array([[0],[1],[1],[0]])

    data = pd.read_csv('./data/data_labeled.csv')
    data = prep.preprocess(data)
    # visualize(data)
    train_set, test_set = prep.split(data)

    X, y = train_set.iloc[:, 1:], train_set.iloc[:, 0]

    # X = numpy_array[:, 1:26]
    # y = numpy_array[:, 0]
    # X_train, X_test = X[:index], X[index:]
    # y_train, y_test = y[:index], y[index:]

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

if __name__ == '__main__':
    main()


# # softmax activation layer : compute values for each sets of scores in x
# # not sure this works