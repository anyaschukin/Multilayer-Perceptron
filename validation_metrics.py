import pandas as pd
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# an auxiliary function that converts probability into class
def probability_to_class(yhat):
    probs = np.copy(yhat)
    probs[probs > 0.5] = 1
    probs[probs <= 0.5] = 0
    return probs

def get_accuracy(Y_hat, Y):
    Y_hat_ = probability_to_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()

def get_validation_metrics(y_pred, y_true):
    # false positives and true positives
    fp = np.sum((y_pred == 1) & (y_true == 0))  # summing the number of examples which fit that particular criteria
    tp = np.sum((y_pred == 1) & (y_true == 1))
    # print("false positives = {}, true positives = {}".format(fp,tp))

    #false negatives and true negatives
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    # print("false negatives = {}, true negatives = {}".format(fn,tn))

    # accuracy = (y_pred == y_true).all(axis=0).mean()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    F1_score = (2 * (precision * recall)) / (precision + recall)

    print("precision = {}\nrecall = {}\nspecificity = {}\nF1_score = {}\n\n".format(precision, recall, specificity, F1_score))

    # return precision, recall, specificity, F1_score