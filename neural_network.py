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

# def backprop(x, y, Wh, Wo, lr):
#     yHat = feed_forward(x, Wh, Wo)

#     # Layer Error
#     Eo = (yHat - y) * relu_prime(Zo)
#     Eh = Eo * Wo * relu_prime(Zh)

#     # Cost derivative for weights
#     dWo = Eo * H
#     dWh = Eh * x

#     # Update weights
#     Wh -= lr * dWh
#     Wo -= lr * dWo

###

## binary cross-entropy loss??
# def entropy_loss(self,y, yhat):
    # nsample = len(y)
    # loss = -1/nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
    # return loss

NUM_NEURONS = 4

# sigmoid?
def activation(x):
    return 1.0/(1+ np.exp(-x))

def activation_derivative(x):
    return x * (1.0 - x)

def compute_loss(y_hat, y):
    return ((y_hat - y)**2).sum()

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1], NUM_NEURONS) 
        self.weights2   = np.random.rand(NUM_NEURONS,1)
        # self.bias1       = np.zeros((self.input.shape[0], NUM_NEURONS))
        self.bias1       = np.zeros((NUM_NEURONS, 1))
        self.bias2       = np.zeros((y.shape[0], 1))
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = activation(np.dot(self.input, self.weights1) + self.bias1)
        self.output = activation(np.dot(self.layer1, self.weights2) + self.bias2) # (layer - 1)

    ## application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    def backprop(self, alpha = 0.1):
        m = self.input.shape[1]

        # output layer
        d_Z2 = self.output - y
        d_weights2 = np.dot(self.layer1, d_Z2)
        # d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * activation_derivative(self.output)))
        d_bias2 = np.sum(d_Z2, axis = 1, keepdims=True) / m

        # hidden layer
        d_Z1 = np.dot(self.weights2.T, d_Z2) * activation_derivative(self.layer1)
        d_weights1 = np.dot(self.input.T, d_Z1) 
        # d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * activation_derivative(self.output), self.weights2.T) * activation_derivative(self.layer1)))
        d_bias1 = np.sum(d_Z2, axis = 1, keepdims=True) / m
       
        ## update the weights with the derivative (slope) of the loss function
        self.weights1 = self.weights1 - alpha * d_weights1
        self.weights2 = self.weights2 - alpha * d_weights2

        self.bias1 = self.bias1 - alpha * d_bias1
        self.bias2 = self.bias2 - alpha * d_bias2

        # d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * activation_derivative(self.output)))
        # d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * activation_derivative(self.output), self.weights2.T) * activation_derivative(self.layer1)))

        # self.weights1 += d_weights1
        # self.weights2 += d_weights2

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


# The ReLufunction performs a threshold
# operation to each input element where values less 
# than zero are set to zero.
# def relu(self,Z):
#     return np.maximum(0, Z)


# # softmax activation layer : compute values for each sets of scores in x
# # not sure this works
# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0) 

# a_layer = theta(weight_l * a_l-1 + b_l)


