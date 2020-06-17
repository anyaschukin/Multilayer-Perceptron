import pandas as pd
import numpy as np
import tools as tools
# import plot as plot
# import visualize as visualize

def split_x_y(data):
    X, y = data.iloc[:, 1:], data.iloc[:, 0]
    
    # transform y into one-hot encoding vector
    target = np.zeros((y.shape[0], 2))
    target[np.arange(y.size),y] = 1
    y = target.T
    
    return X, y

def split(data):
    # remove feature names
    new_header = data.iloc[0]
    data = data[1:]
    data.columns = new_header

    # shuffle and split data into train and test sets
    shuffled_data = data.sample(frac=1)
    i = int(0.7 * len(data))
    train_set = shuffled_data[:i]
    test_set = shuffled_data[i:]

    # X = numpy_array[:, 1:26]
    # y = numpy_array[:, 0]
    # X_train, X_test = X[:index], X[index:]
    # y_train, y_test = y[:index], y[index:]

    return train_set, test_set

def scale(data, scaling='standardize'):
    normed = data
    if scaling == 'normalize':
        # forcing the mean of each measurement to 0, and dividing each measurement by the maximum value of that measurement in the dataset
        normed =(normed-normed.mean())/normed.max()
    if scaling == 'min_max_normalize':
        # all data values are adjusted to lie on a bell curve btwn 0 and 10, starting at zero
        normed = (normed - normed.min())/(normed.max()-normed.min())
    if scaling == 'standardize':
        # all data values are centered around the mean (zero) with a unit standard deviation (min, max)
        normed = (normed-normed.mean())/normed.std()
        # normed=(normed-normed.mean())/(normed.max()-normed.mean()) # for numbers btwn 1 and -1
    normed["diagnosis"] = data["diagnosis"]
    return normed
    
# remove [unnecessary columns] and [columns with NaN] and scale data
def preprocess(data):
    # see if there are any columns with missing/null data
    # print(data.isnull().sum())
    try:
        data = data.drop(columns=['id', 'Unnamed: 32'])
        data = data.dropna()
        data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    except Exception:
        tools.error_exit('Failed to preprocess data. Is data valid?')
    data = scale(data)
    return data

## To use, add the following to your main:
# data = pd.read_csv('./data/data_labeled.csv')

# to see your data
# print(data)
# print(data.describe())

# data = preprocess(data)

# features = visualize.select_feature(data)
# visualize.pair_plot(features)

# data = data.drop(columns=['diagnosis'])
# train_set, test_set = split(data)
# X, y = train_set.iloc[:, 1:], train_set.iloc[:, 0]
