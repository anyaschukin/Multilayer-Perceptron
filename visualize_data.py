import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import preprocessing
import numpy as np
import tools as tools

# data = pd.read_csv('./data 2.csv')

# if isinstance(data, pd.DataFrame):
#     print("I am a dataframe")
# else:
#     print("fuck you")

# print(data.head())

def pairplot(data):
    radius = data[['radius_mean','radius_se','radius_worst','diagnosis']]
    sns.pairplot(radius, hue='diagnosis',palette="husl", markers=["o", "s"],height=4)
    plt.show()
    # try:
    # except Exception:
        # tools.error_exit('Failed to visualize data. Is data valid?')

def scale(data):
    for feature in data:
        value_min = data[feature].min()
        value_max = data[feature].max()
        # print("min = {}, max = {}".format(value_min, value_max))
        # change to: if mean > 10 ?       
        if value_max > 10:
            for value in data[feature]:
                # print(value)
                std_value = (value - value_min) / (value_max - value_min)
                data[feature].replace(to_replace=value, value=std_value, inplace=True)
    # print(data)
    
# remove 1) unnecessary columns and 2) columns with NaN
def preprocess(data):
    # see if there are any columns with missing/null data
    # print(data.isnull().sum())
    try:
        data = data.drop(columns=['id', 'Unnamed: 32'])
        data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    except Exception:
        tools.error_exit('Failed to preprocess data. Is data valid?')
    # data = scale(data)
    # normalize here?
    return data

def main():
    data = pd.read_csv('./data/data_labeled.csv')
    # print(data)
    # print(data.describe())
    data = preprocess(data)
    pairplot(data)

if __name__ == '__main__':
    main()
