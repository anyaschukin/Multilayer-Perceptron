import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tools as tools

def pair_plot(features):
    try:
        sns.pairplot(features, hue='diagnosis',palette="husl", markers=["o", "s"],height=4)
        plt.show()
    except Exception:
        tools.error_exit('Failed to visualize data. Is data valid?')

def heat_map(data):
    mean = data[['texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','diagnosis']]
    plt.figure(figsize=(15,10))
    heat = sns.heatmap(mean.corr(), vmax=1, square=True, annot=True, cmap="YlGnBu")
    plt.show()

def strip_plot(data):
    data_strip = data.drop('diagnosis',axis=1)
    data_strip = data[['texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean']]
    for index,columns in enumerate(data_strip):
        plt.figure(figsize=(15,10))
        sns.stripplot(x='diagnosis', y= columns, data= data, jitter=True, palette = 'Set1')
        plt.title('Diagnosis vs ' + str(columns))
        plt.show()