import pandas as pd
import numpy as np
import tools as tools
import matplotlib.pyplot as plt
import preprocess as prep



# an auxiliary function that converts probability into class
def probability_to_class(yhat):
    probs = np.copy(yhat)
    probs[probs > 0.5] = 1
    probs[probs <= 0.5] = 0
    return probs

def get_accuracy(Y_hat, Y):
    Y_hat_ = probability_to_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


LAYER1_NEURONS = 16
LAYER2_NEURONS = 16
LAYER3_NEURONS = 16
LAYER4_NEURONS = 2

# The ReLufunction performs a threshold operation to each input element 
# where values less than zero are set to zero.
def ReLU(z):
    return np.maximum(0, z)

def ReLU_prime(z):
    return 1 * (z > 0)
    # return 1 if z > 0 else 0

def leaky_ReLU(z, alpha = 0.01):
	return np.where(z >= 0, z, z * alpha)
    # return max(alpha * z, z)

def leaky_ReLU_prime(z, alpha = 0.01):
    return np.where(z >= 0, 1, alpha)
	# return 1 if x > 0 else alpha

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
    # Numerically Stable: (z - np.max(z) shifts the values of z so that the highest number is 0... [1, 3, 5] -> [-4, -2, 0]
    z_max = np.max(z, axis=0, keepdims=True)
    e = np.exp(z - z_max)
    # e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

def softmax_prime(z):
    return softmax(z) * (1-softmax(z))

epsilon = 1e-2  

# binary cross-entropy loss
def compute_loss(yhat, y):
    m = len(y)
    loss = -1/m * (np.sum(np.dot(np.log(yhat).T, y) + np.dot((1 - y).T, np.log(1 - yhat))))
    return loss

def compute_loss_prime(yhat, y):
    d_loss = - (np.divide(y, yhat) - np.divide(1 - y, 1 - yhat)) ## not sure if this should be - or +, according to Kaggle example
    return d_loss

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(LAYER1_NEURONS, self.input.shape[1]) * np.sqrt(2/self.input.shape[1]) ## * 0.01 # self.input.shape[1] is num_features
        self.weights2   = np.random.rand(LAYER2_NEURONS, LAYER1_NEURONS) * np.sqrt(2/LAYER1_NEURONS) ## * 0.01
        self.weights3   = np.random.rand(LAYER3_NEURONS, LAYER2_NEURONS) * np.sqrt(2/LAYER2_NEURONS) ## * 0.01
        self.weights4   = np.random.rand(2, LAYER3_NEURONS) * np.sqrt(2/LAYER3_NEURONS) ## * 0.01  # if multiple classification, it should (NUM_NEURONS, output_size(# of classes))

        self.bias1       = np.zeros((LAYER1_NEURONS, 1)) # 4 x 1
        self.bias2       = np.zeros((LAYER2_NEURONS, 1))
        self.bias3       = np.zeros((LAYER3_NEURONS, 1))
        self.bias4       = np.zeros((2, 1)) # (1, num_of_classes)... maybe last layer shouldn't have bias
       
        # self.y           = y.to_numpy().reshape(y.shape[0], 1)
        self.y           = y
        self.output     = np.zeros((2, self.y.shape[0]))

    def feedforward(self, activation = softmax, activation_hidden = sigmoid):
        
        self.Z1 = np.dot(self.weights1, self.input.T) + self.bias1
        self.layer1 = activation_hidden(self.Z1) 
        self.Z2 = np.dot(self.weights2, self.layer1) + self.bias2
        self.layer2 = activation_hidden(self.Z2)
        self.Z3 = np.dot(self.weights3, self.layer2) + self.bias3
        self.layer3 = activation_hidden(self.Z3)
        self.Z4 = np.dot(self.weights4, self.layer3) + self.bias4 # layer = theta(weight_l * a_l-1 + b_l)
        # print("Z4 \n{}".format(self.Z4))
        self.output = activation(self.Z4) # layer = theta(weight_l * a_l-1 + b_l)
        # print("output =  \n{}\n target = \n{}".format(self.output, self.y))


    ## application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
    def backprop(self, d_activation = softmax_prime, d_activation_hidden = sigmoid_prime, learning_rate = 0.01):
        m = self.input.shape[0] # num examples or batch size
        # weights and d_weights should have the same dimensions

        # d_Z4 = d_A4 * d_activation(self.output) # not sure if we need activation derivative on output
        # d_A4 = -self.y/self.output + (1-self.y)/(1-self.output) # from Kaggle dataset

        d_A4 = compute_loss_prime(self.output, self.y)
        # d_A4 = self.output - self.y

        # output layer
        d_Z4 = d_A4 * d_activation(self.Z4) # not sure if we need activation derivative on output
        d_weights4 = np.dot(d_Z4, self.layer3.T)
        d_bias4 = np.sum(d_Z4, axis = 1, keepdims=True) # should be either axis 0 or 1, should create shape of 1,1 
        d_A3 = np.dot(self.weights4.T, d_Z4)

        # hidden layers
        # d_Z3 = d_A3 * d_activation_hidden(self.layer3)
        d_Z3 = d_A3 * d_activation_hidden(self.Z3)
        d_weights3 = np.dot(d_Z3, self.layer2.T)  # (layer-1) * output error
        d_bias3 = np.sum(d_Z3, axis = 1, keepdims=True)
        d_A2 = np.dot(self.weights3.T, d_Z3)
 
        # d_Z2 = d_A2 * d_activation_hidden(self.layer2)
        d_Z2 = d_A2 * d_activation_hidden(self.Z2)
        d_weights2 = np.dot(d_Z2, self.layer1.T) # (layer-1) * output error
        d_bias2 = np.sum(d_Z2, axis = 1, keepdims=True) 
        d_A1 = np.dot(self.weights2.T, d_Z2)

        # d_Z1 = d_A1 * d_activation_hidden(self.layer1)
        d_Z1 = d_A1 * d_activation_hidden(self.Z1)
        d_weights1 = np.dot(d_Z1, self.input) # (layer-1) * output error
        d_bias1 = np.sum(d_Z2, axis = 1, keepdims=True)

       
        ## update the weights with the derivative (slope) of the loss function (SGD)
        self.weights1 -= learning_rate * (d_weights1 / m) 
        self.weights2 -= learning_rate * (d_weights2 / m)
        self.weights3 -= learning_rate * (d_weights3 / m)
        self.weights4 -= learning_rate * (d_weights4 / m)

        self.bias1 -= learning_rate * (d_bias1 / m)
        self.bias2 -= learning_rate * (d_bias2 / m)
        self.bias3 -= learning_rate * (d_bias3 / m)
        self.bias4 -= learning_rate * (d_bias4 / m)


def main():

    data = pd.read_csv('./data/data_labeled.csv')
    data = prep.preprocess(data)
    # visualize(data)
    train_set, test_set = prep.split(data)

    # X, y = train_set.iloc[:, 1:], train_set.iloc[:, 0]
    X, y = train_set.iloc[:, 1:], train_set.iloc[:, 0]

    # print("y \n{}\n".format(y))

    # transform y into one-hot encoding vector
    target = np.zeros((y.shape[0], 2))
    target[np.arange(y.size),y] = 1
    y = target.T

    # for row in y:
        # if row == 1: #if malignant
            # target[row] = [0,1]
        # if row == 0: #if benign (can only be labeled 'B' or 'M')
            # target[row] = [1,0]
    
    print("target \n{}\n".format(target))

    # y = y.reshape(y.shape[0], 1)
    # y = y.to_numpy().shape[0]

    # X = numpy_array[:, 1:26]
    # y = numpy_array[:, 0]
    # X_train, X_test = X[:index], X[index:]
    # y_train, y_test = y[:index], y[index:]

    nn = NeuralNetwork(X, y)
    loss_values = []

    for i in range(20000):
        nn.feedforward()
        # if i == 1000:
            # plt.scatter(nn.y, nn.output)
            # plt.show()
        nn.backprop()
        loss = compute_loss(nn.output, nn.y)
        # print("output = {}".format(nn.output))
        # print("loss = {}, i = {}".format(loss, i))
        loss_values.append(loss)

    # print(nn.output.shape)
    # print(nn.y.shape)

    print(nn.output.T)
    print("accuracy = {}".format(get_accuracy(nn.output, nn.y)))
    # print(f" final loss : {loss}")

    # plt.plot(loss_values)
    # plt.show()

if __name__ == '__main__':
    main()


# # softmax activation layer : compute values for each sets of scores in x
# # not sure this works