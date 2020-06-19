import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import numpy as np
import tools as tools
import plot as plot


## this file is DEPRECATED and has been REPLACED by preprocess.py and visualize.py


def select_feature(data):

	mean = data[['texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','diagnosis']]
	# se = data[['texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','diagnosis']]
	# worst = data[['texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst','diagnosis']]
	##
	# radius = data[['radius_mean','radius_se','radius_worst','diagnosis']]
	# texture = data[['texture_mean','texture_se','texture_worst','diagnosis']]
	# perimeter = data[['perimeter_mean','perimeter_se','perimeter_worst','diagnosis']]
	# area = data[['area_mean','area_se','area_worst','diagnosis']]
	# smoothness = data[['smoothness_mean','smoothness_se','smoothness_worst','diagnosis']]
	# compactness = data[['compactness_mean','compactness_se','compactness_worst','diagnosis']]
	# concavity = data[['concavity_mean','concavity_se','concavity_worst','diagnosis']]
	# concave_points = data[['concave points_mean','concave points_se','concave points_worst','diagnosis']]
	# symmetry = data[['symmetry_mean','symmetry_se','symmetry_worst','diagnosis']]
	# fractal_dimension = data[['fractal_dimension_mean','fractal_dimension_se','fractal_dimension_worst','diagnosis']]

	features = mean
	return features

def scale(data):
	for feature in data:
		value_min = data[feature].min()
		value_max = data[feature].max()
		# change to: if mean > 10 ?	   
		if value_max > 10:
			for value in data[feature]:
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

	## for visualizing data
	features = select_feature(data)
	plot.pair_plot(features)	# can select by feature for pairplot
	# plot.heat_map(data)	   # only mean
	# plot.strip_plot(data)	 # only mean

if __name__ == '__main__':
	main()




#####

# if isinstance(data, pd.DataFrame):
#	 print("I am a dataframe")
# else:
#	 print("fuck you")

# print(data.head())