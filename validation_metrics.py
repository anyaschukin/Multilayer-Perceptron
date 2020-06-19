import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class colors:
    BLUE = "\033[94m"
    GREEN = "\033[32m"
    LGREEN = "\033[92m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    ENDC = "\033[0m"

# an auxiliary function that converts probability into class
def probability_to_class(yhat):
    probs = np.copy(yhat)
    probs[probs > 0.5] = 1
    probs[probs <= 0.5] = 0
    return probs

def get_accuracy(y_pred, y_true):
    y_pred_ = probability_to_class(y_true)
    return (y_pred_ == y_true).all().mean()

def get_validation_metrics(y_pred, y_true):
    # false positives and true positives
    fp = np.sum((y_pred == 1) & (y_true == 0))  # summing the number of examples which fit that particular criteria
    tp = np.sum((y_pred == 1) & (y_true == 1))
    # print("false positives = {}, true positives = {}".format(fp,tp))

    #false negatives and true negatives
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    # print("false negatives = {}, true negatives = {}".format(fn,tn))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    F1_score = (2 * (precision * recall)) / (precision + recall)

    print("\n" + colors.BLUE + "Accuracy"  + colors.ENDC + " = {}".format(accuracy))
    print(colors.BLUE + "Accuracy is the proportion of predictions that the model classified correctly.\n" + colors.ENDC)
    
    print(colors.CYAN + "Precision" + colors.ENDC + " = {}".format(precision))
    print(colors.CYAN + "Precision answers the question 'What proportion of positive identifications was actually correct?'\n" + colors.ENDC)
    
    print(colors.GREEN + "Recall" + colors.ENDC + " = {}".format(recall))
    print(colors.GREEN + "Recall aka True Positive Rate, answers the question 'What proportion of actual positives was identified correctly?'\n" + colors.ENDC)
    
    print(colors.YELLOW + "Specificity" + colors.ENDC + " = {}".format(specificity))
    print(colors.YELLOW + "Specificity aka True Negative Rate, answers the question 'What proportion of actual negatives was identified correctly?'\n" + colors.ENDC)
    
    print(colors.MAGENTA + "F1_score" + colors.ENDC + " = {}".format(F1_score))
    print(colors.MAGENTA + "F1 Score is the harmonic mean of precision and recall.\nIt can have a maximum score of 1 (perfect precision and recall) and a minimum of 0.\nOverall, it is a measure of the preciseness and robustness of your model.\n" + colors.ENDC)

def plot_learning(train_losses, test_losses):

    # Draw lines
    plt.plot(train_losses, label="Training loss", color='c')
    plt.plot(test_losses, label="Test/Validation loss", color='b')

    plt.title("Learning Curve")
    plt.xlabel("Epochs"), plt.ylabel("Loss"), plt.legend(loc="best")
    plt.legend()
    plt.show()