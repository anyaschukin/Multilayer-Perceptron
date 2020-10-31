import os
import sys
import argparse

def error_exit(err_msg):
    print('Error: {}' .format(err_msg))
    sys.exit()

def is_file(path):
    if not os.path.isfile(path):
        error_exit('path specified ({}) does not exist' .format(path))

class Args:
	def __init__(self, parser):
		self.dataset = parser.Dataset
		self.data_visualize = parser.data_visualize
		self.train = parser.train
		self.predict = parser.predict
		self.mini_batch = parser.mini_batch
		self.evaluation = parser.evaluation

def get_args():
	parser = argparse.ArgumentParser(description="Multilayer Perceptron predicts whether a cancer is malignant or benign.")
	parser.add_argument('Dataset',
					   metavar='dataset',
					   type=str,
					   help='provide a valid dataset.')
	parser.add_argument('-b',
						'--mini_batch',
						action='store_true',
						help='train neural network on mini-batch, of size 32')
	parser.add_argument('-e',
						'--evaluation',
						action='store_true',
						help='display in-depth learning evaluation metrics')
	parser.add_argument('-model',
					   metavar='model',
					   type=str,
					   help='provide a valid model.')
	parser.add_argument('-p',
						'--predict',
						action='store_true',
						help='load a saved model and perform a prediction on a given dataset')
	parser.add_argument('-s',
						'--save_model',
						action='store_true',
						help='save a trained model')
	parser.add_argument('-t',
						'--train',
						action='store_true',
						help='use backpropagation and gradient descent to learn on the training dataset and save the model')
	parser.add_argument('-v',
						'--visualize_data',
						action='store_true',
						help='display data with graphs')

	args = parser.parse_args()

	if not (args.visualize_data or args.train or args.predict):
		parser.error("Error: must use either 'visualize' or 'train' or 'predict' option.")
	# arguments = parser.parse_args()
	# args = Args(arguments)
	# dataset = args.Dataset
	# data_visualize = args.data_visualize
	# train = args.train
	# predict = args.predict
	# mini_batch = args.mini_batch
	# evaluation = args.evaluation
	# return dataset, data_visualize, train, predict, mini_batch, evaluation
	return args