import pandas as pd
import numpy as np
import tools as tools

# split data into X and y
def split_x_y(data):
	X, y = data.iloc[:, 1:], data.iloc[:, 0]
	
	# transform y into one-hot encoding vector
	target = np.zeros((y.shape[0], 2))
	target[np.arange(y.size),y] = 1
	y = target.T
	
	return X, y

# shuffle and split data into train and test sets
def split(data):
	# remove feature names
	new_header = data.iloc[0]
	data = data[1:]
	data.columns = new_header

	shuffled_data = data.sample(frac=1)
	i = int(0.7 * len(data))
	train_set = shuffled_data[:i]
	test_set = shuffled_data[i:]

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
def preprocess(args):

	# check if there are any columns with missing/null data
	# print(data.isnull().sum())

	try:
		data = pd.read_csv(args.Dataset)
		data.columns = ['id','diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']
		data = data.drop(columns=['id'])
		# print(len(data.columns))
		data = data.dropna()
		data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
	except Exception:
		tools.error_exit('Failed to preprocess data. Is data valid?')
	data = scale(data)

	return data