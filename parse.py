import argparse

def parse_args():
	my_parser = argparse.ArgumentParser(description="Multilayer Perceptron predicts whether a cancer is malignant or benign.")
	my_parser.add_argument('Dataset',
					   metavar='dataset',
					   type=str,
					   help='provide a valid dataset.')
	my_parser.add_argument('-d',
						'--data_visualize',
						action='store_true',
						help='display data with graphs')
	my_parser.add_argument('-t',
						'--train',
						action='store_true',
						help='use backpropagation and gradient descent to learn on the training dataset and save the model')
	my_parser.add_argument('-p',
						'--predict',
						action='store_true',
						help='perform a prediction on a given set')
	my_parser.add_argument('-m',
						'--mini_batch',
						action='store_true',
						help='train neural network on mini-batch, of size 32')
	my_parser.add_argument('-e',
						'--evaluation',
						action='store_true',
						help='display in-depth learning evaluation metrics')
	my_parser.add_argument('-l',
						'--learning',
						action='store_true',
						help='quiet mode: don\'t display learning curve graph')
	args = my_parser.parse_args()
	dataset = args.Dataset
	data_visualize = args.data_visualize
	train = args.train
	predict = args.predict
	mini_batch = args.mini_batch
	evaluation = args.evaluation
	learning = args.learning
	return dataset, data_visualize, train, predict, mini_batch, evaluation, learning