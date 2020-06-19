import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
# import numpy as np
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

def visualize(data):
	print(data)
	time.sleep(3)
	print(data.describe())
	time.sleep(3)

	features = select_feature(data)
	pair_plot(features)	 # can select by feature for pairplot
	heat_map(data)		  # only mean
	strip_plot(data)		# only mean