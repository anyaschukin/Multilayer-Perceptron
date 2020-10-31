import sys
from tools import get_args
from preprocess import *
from visualize import visualize
from neural_network import train_model

def main():
	# try:
	args = get_args()

	data = pd.read_csv(args.Dataset)
	data = preprocess(data)

	if args.data_visualize:
		visualize(data)
		sys.exit(1) 

	# train_model(data, train, predict, mini_batch, evaluation)
	train_model(data, args)

	# except:
		# print("\nError. You did something wrong.\n")
		# pass

if __name__ == '__main__':
	main()