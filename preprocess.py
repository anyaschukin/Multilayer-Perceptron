import pandas as pd
# import numpy as np
import tools as tools
# import plot as plot
# import visualize as visualize

def scale(data, scaling='min_max_normalization'):
    print(data)
    normed = data
    if scaling == 'min_max_normalization':
        # all data values are adjusted to lie on a bell curve btwn 0 and 10, starting at zero
        normed = (normed - normed.min())/(normed.max()-normed.min())*10
    if scaling == 'standardization':
        # all data values are centered around the mean (zero) with a unit standard deviation (min, max)
        normed=(normed-normed.mean())/normed.std()*10
    normed["diagnosis"] = data["diagnosis"]
    return normed
    
# remove [unnecessary columns] and [columns with NaN]
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
# features = visualize.select_feature(data)
# visualize.pair_plot(features)
