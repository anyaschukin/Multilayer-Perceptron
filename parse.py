import argparse

def parse_args():
	my_parser = argparse.ArgumentParser(description="Multilayer Perceptron predicts whether a cancer is malignant or benign.")
	my_parser.add_argument('Filepath',
					   metavar='filepath',
					   type=str,
					   help='provide a valid dataset.')
    my_parser.add_argument('-d',
                        '--data_visualize',
                        action='store_true',
                        help='display data with graphs')
	my_parser.add_argument('-b',
                        '--batch',
                        action='store_true',
                        help='display graph of facts and rules nodes')
	my_parser.add_argument('-t',
                        '--train',
                        action='store_true',
                        help='use backpropagation and gradient descent to learn on the training dataset and save the model')
	my_parser.add_argument('-p',
                        '--predict',
                        action='store_true',
                        help='perform a prediction on a given set')
	my_parser.add_argument('-m',
                        '--metrics',
                        action='store_true',
                        help='display in-depth learning evaluation metrics')
    my_parser.add_argument('-l',
                        '--learning_visualize',
                        action='store_true',
                        help='display extra evaluation metrics')
	args = my_parser.parse_args()
	filepath = args.Filepath
	data_visualize = args.data_visualize
	batch = args.batch
	train = args.train
	predict = args.predict
	metrics = args.metrics
	learning_visualize = args.learning_visualize
	return filepath, graph, color, timer, logic
