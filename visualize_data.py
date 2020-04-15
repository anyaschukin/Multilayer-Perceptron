import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn import preprocessing
import numpy as np

# data = pd.read_csv('./data 2.csv')

# if isinstance(data, pd.DataFrame):
#     print("I am a dataframe")
# else:
#     print("fuck you")

# print(data.head())


def preprocess(data):
    try:
        data = data.drop(columns=['id', 'Unnamed: 32'])
        data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    except Exception:
        tools.error_exit('Failed to preprocess data. Is data valid?')
    return data

def main():
    data = pd.read_csv('./data_labeled2.csv')
    data = preprocess(data)
    print(data)
    print(data.describe())

if __name__ == '__main__':
    main()
