import pandas as pd
import numpy as np
import tools as tools
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from preprocess import *
from activations import *
from validation_metrics import *

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import precision_score

LAYER1_NEURONS = 16
LAYER2_NEURONS = 16
LAYER3_NEURONS = 16
LAYER4_NEURONS = 2

class NeuralNetwork:
    def __init__(self, num_features, batch_size, learning_rate):
        self.batch_size     = batch_size
        self.learning_rate  = learning_rate
        self.input          = None
        self.weights1       = np.random.rand(LAYER1_NEURONS, num_features) * np.sqrt(2/num_features) ## * 0.01
        self.weights2       = np.random.rand(LAYER2_NEURONS, LAYER1_NEURONS) * np.sqrt(2/LAYER1_NEURONS) ## * 0.01
        self.weights3       = np.random.rand(LAYER3_NEURONS, LAYER2_NEURONS) * np.sqrt(2/LAYER2_NEURONS) ## * 0.01
        self.weights4       = np.random.rand(2, LAYER3_NEURONS) * np.sqrt(2/LAYER3_NEURONS) ## * 0.01  # if multiple classification, it should (NUM_NEURONS, output_size(# of classes))

        self.bias1          = np.zeros((LAYER1_NEURONS, 1))
        self.bias2          = np.zeros((LAYER2_NEURONS, 1))
        self.bias3          = np.zeros((LAYER3_NEURONS, 1))
        self.bias4          = np.zeros((2, 1)) # (num_of_classes, 1)... maybe last layer shouldn't have bias
       
        self.y              = None
        self.output         = np.zeros((2, batch_size))

    def feedforward(self, activation = softmax, activation_hidden = sigmoid):
        
        self.Z1 = np.dot(self.weights1, self.input.T) + self.bias1
        self.layer1 = activation_hidden(self.Z1) 

        self.Z2 = np.dot(self.weights2, self.layer1) + self.bias2
        self.layer2 = activation_hidden(self.Z2)
        
        self.Z3 = np.dot(self.weights3, self.layer2) + self.bias3
        self.layer3 = activation_hidden(self.Z3)

        self.Z4 = np.dot(self.weights4, self.layer3) + self.bias4 # layer = theta(weight_l * a_l-1 + b_l)
        self.output = activation(self.Z4) # layer = theta(weight_l * a_l-1 + b_l)


    ## application of the chain rule to find derivative of the loss function with respect to weights and bias
    def backprop(self, d_activation = softmax_prime, d_activation_hidden = sigmoid_prime):

        m = self.batch_size # num examples or batch size

        d_A4 = compute_loss_prime(self.output, self.y)

        # output layer
        d_Z4 = d_A4 * d_activation(self.Z4)
        d_weights4 = np.dot(d_Z4, self.layer3.T) # weights and d_weights should have the same dimensions
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

    def predict(self, test_set):
        num_examples = test_set.shape[0]

        shuffle(test_set)
        self.input, self.y = split_x_y(test_set)
        self.output = np.zeros((2, num_examples))

        # replicate feedforward for testing
        self.feedforward()
        # accuracy = get_accuracy(self.output, self.y)
        # print(accuracy)
        # return self.output.T

def main():

    data = pd.read_csv('./data/data_labeled.csv')
    data = preprocess(data)
    # visualize(data)
    train_set, test_set = split(data)
    
    num_examples = train_set.shape[0]
    num_features = train_set.shape[1] - 1
    
    learning_rate = 0.01
    batches = 'mini_batch'

    if batches == 'SGD':
        batch_size = 1
        epochs = 40000
    elif batches == 'mini_batch':
        batch_size = 32 #64
        epochs = 1500
    elif batches == 'whole_batch':         
        batch_size = num_examples
        epochs = 20000
    else:
        batch_size = num_examples
        epochs = 20000

    prediction_program = True
    training_program = True

    nn = NeuralNetwork(num_features, batch_size, learning_rate)
    test_losses, train_losses = [], []

    for epoch in range(epochs):
        shuffle(train_set)
        for i in range(0, num_examples, nn.batch_size):
            nn.input, nn.y = split_x_y(train_set[i:i+batch_size])
            nn.feedforward()
            nn.backprop()
            
            train_loss = compute_loss(nn.output[:, 0], nn.y[:, 0])
            train_losses.append(train_loss)

            nn.predict(test_set)
            
            test_loss = compute_loss(nn.output[:, 0], nn.y[:, 0])
            test_losses.append(test_loss)

        # print("epoch {}/{}: train loss = {}, test loss = {}".format(epoch, epochs, round(train_loss, 4), round(test_loss, 4)))
    
    if training_program == True:
        # save network params
        print("hello\n")
    if prediction_program == True:
        print("colors.OKGREEN + Final loss on validation set = {}\n".format(test_loss) + colors.ENDC)
    
    # plot both train and test loss
    # plt.plot(loss_values)
    # plt.show()       

    y_pred = probability_to_class(nn.output.T)
    get_validation_metrics(y_pred[:, 0], nn.y.T[:, 0])
    define_validation_metrics()

if __name__ == '__main__':
    main()



# Confusion Matrix
# confusion_matrix(y_true, y_pred,  labels=['malignant', 'benign'])
# Accuracy
# accuracy_score(y_true, y_pred, labels)
# Recall
# recall_score(y_true, y_pred, average=None)
# Precision
# precision_score(y_true, y_pred, average=None)

# y = y.reshape(y.shape[0], 1)
# y = y.to_numpy().shape[0]
# X = numpy_array[:, 1:26]
# y = numpy_array[:, 0]
# X_train, X_test = X[:index], X[index:]
# y_train, y_test = y[:index], y[index:]