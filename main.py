import sys
from tools import parse_args
from preprocess import *
from visualize import visualize
from neural_network import train_model

def main():
	# try:
	dataset, data_visualize, train, predict, mini_batch, evaluation = parse_args()

	data = pd.read_csv(dataset)
	data = preprocess(data)

	if data_visualize:
		visualize(data)
		sys.exit(1) 

	train_model(data, train, predict, mini_batch, evaluation)

	# except:
		# print("\nError. You did something wrong.\n")
		# pass

if __name__ == '__main__':
	main()