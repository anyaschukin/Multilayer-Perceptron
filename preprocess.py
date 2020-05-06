import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
import tools as tools
# import plot as plot

def scale(data):
    print(data)
    normed = data
    # all data values are adjusted to lie on a bell curve btwn 0 and 10 (that's the * by 10)
    normed = (normed - normed.min())/(normed.max()-normed.min())*10

    # put data on a bell curve, centered on zero (with max and min on either side)
    # normed=(normed-normed.mean())/normed.std()*10
    normed["diagnosis"] = data["diagnosis"]

    # normed = normed.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x))*10, axis = 0)

    print(normed)
    
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
    # normalize here?
    return data


## To use, add the following to your main:
data = pd.read_csv('./data/data_labeled.csv')

# to see your data
# print(data)
# print(data.describe())

data = preprocess(data)
# print(data)

# frame.apply(f, axis=1) where f is a function that does something with a row