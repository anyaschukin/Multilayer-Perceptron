import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import tools as tools

class colors:
	GREEN = "\033[32m"
	ENDC = "\033[0m"

def pair_plot(feature):
	try:
		sns.pairplot(feature, hue='diagnosis',palette="husl", markers=["o", "s"],height=4)
		plt.show()
	except Exception:
		tools.error_exit('Failed to visualize data. Is data valid?')

def heat_map(feature):
	plt.figure(figsize=(15,10))
	heat = sns.heatmap(feature.corr(), vmax=1, square=True, annot=True, cmap="YlGnBu")
	plt.show()

def strip_plot(feature):
	data_strip = feature.drop('diagnosis',axis=1)
	data_strip = feature[feature.columns.tolist()]
	for index,columns in enumerate(data_strip):
		plt.figure(figsize=(15,10))
		sns.stripplot(x='diagnosis', y= columns, data= feature, jitter=True, palette = 'Set1')
		plt.title('Diagnosis vs ' + str(columns))
		plt.show()

def select_feature(data):
	options = ["mean", "se", "worst", "radius", "texture", "perimeter", "area", "smoothness","compactness", "concavity", "concave points", "symmetry", "fractal dimension"]

	for i in range(len(options)):
		print(str(i) + ":", options[i])
		# print(str(i+1) + ":", options[i])

	while True:
		choice = input("Enter a number: ")
		try:
			choice = int(choice)
			if choice > -1 and choice < 13:
				choice = options[int(choice)]
				break
			else:
				print(colors.GREEN + "Invalid input! Let's try that again." + colors.ENDC)
				pass
		except:
			print(colors.GREEN + "Invalid input! Let's try that again." + colors.ENDC)
			pass	

	mean = data[['texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','diagnosis']]
	se = data[['texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','diagnosis']]
	worst = data[['texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst','diagnosis']]
	radius = data[['radius_mean','radius_se','radius_worst','diagnosis']]
	texture = data[['texture_mean','texture_se','texture_worst','diagnosis']]
	perimeter = data[['perimeter_mean','perimeter_se','perimeter_worst','diagnosis']]
	area = data[['area_mean','area_se','area_worst','diagnosis']]
	smoothness = data[['smoothness_mean','smoothness_se','smoothness_worst','diagnosis']]
	compactness = data[['compactness_mean','compactness_se','compactness_worst','diagnosis']]
	concavity = data[['concavity_mean','concavity_se','concavity_worst','diagnosis']]
	concave_points = data[['concave points_mean','concave points_se','concave points_worst','diagnosis']]
	symmetry = data[['symmetry_mean','symmetry_se','symmetry_worst','diagnosis']]
	fractal_dimension = data[['fractal_dimension_mean','fractal_dimension_se','fractal_dimension_worst','diagnosis']]

	features = {"mean": mean, "se": se, "worst": worst, "radius": radius, "texture": texture, "perimeter": perimeter, "area": area, "smoothness": smoothness,"compactness": compactness, "concavity": concavity, "concave points": concave_points, "symmetry": symmetry, "fractal dimension": fractal_dimension}
	return features[choice]

def visualize(data):
	print(data)
	input("\n\nPress Enter to continue...\n\n")
	print(data.describe())
	input("\n\nPress Enter to continue...\n\n")

	feature = select_feature(data)
	pair_plot(feature)
	heat_map(feature)
	strip_plot(feature)
