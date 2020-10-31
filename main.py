import sys
import json
from tools import get_args
from preprocess import preprocess
from visualize import visualize
from neural_network import *

def main():
	# try:
	args = get_args()

	# data = pd.read_csv(args.Dataset)
	data = preprocess(args)

	if args.visualize_data:
		visualize(data)
		sys.exit(1) 

	###########
	train_set, test_set = split(data)
	
	num_examples = train_set.shape[0]
	num_features = train_set.shape[1] - 1
	if args.mini_batch:
		batch_size = 32		# or 64
		epochs = 1500
	else:
		batch_size = num_examples
		epochs = 20000

	nn = NeuralNetwork(num_features, batch_size, epochs)

	if args.train:
		nn.train(data, args, train_set, test_set, num_examples)

	if args.predict:
		with open(args.model) as file:
			model = json.load(file)
		
		nn.load_model(model)
		nn.predict(test_set)


	# except:
		# print("\nError. You did something wrong.\n")
		# pass

if __name__ == '__main__':
	main()