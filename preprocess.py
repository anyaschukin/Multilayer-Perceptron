import pandas as pd
# import numpy as np
import tools as tools
# import plot as plot
# import visualize as visualize

def split(data):
    shuffled_data = data.sample(frac=1)
    i = int(0.7 * len(data))
    train_set = shuffled_data[:i]
    test_set = shuffled_data[i:]
    return train_set, test_set

    # X = numpy_array[:, 1:26]
    # y = numpy_array[:, 0]
    # X_train, X_test = X[:index], X[index:]
    # y_train, y_test = y[:index], y[index:]

def scale(data, scaling='min_max_normalize'):
    print(data)
    normed = data
    if scaling == 'min_max_normalize':
        # all data values are adjusted to lie on a bell curve btwn 0 and 10, starting at zero
        normed = (normed - normed.min())/(normed.max()-normed.min())*10
    if scaling == 'standardize':
        # all data values are centered around the mean (zero) with a unit standard deviation (min, max)
        normed=(normed-normed.mean())/normed.std()*10
    normed["diagnosis"] = data["diagnosis"]
    return normed
    
# remove [unnecessary columns] and [columns with NaN] and scale data
def preprocess(data):
    # see if there are any columns with missing/null data
    # print(data.isnull().sum())
    try:
        data = data.drop(columns=['id', 'Unnamed: 32'])
        data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    except Exception:
        tools.error_exit('Failed to preprocess data. Is data valid?')
    data = scale(data)
    return data

## To use, add the following to your main:
data = pd.read_csv('./data/data_labeled.csv')

# to see your data
# print(data)
# print(data.describe())

data = preprocess(data)
train_set, test_set = split(data)
print(train_set)
# features = visualize.select_feature(data)
# visualize.pair_plot(features)
