import os
import sys

def error_exit(err_msg):
	print('Error: {}' .format(err_msg))
	sys.exit()

def is_file(path):
	if not os.path.isfile(path):
		error_exit('path specified ({}) does not exist' .format(path))

def parse_args(usage):
	my_parser = argparse.ArgumentParser(description=usage)
	my_parser.add_argument('Dataset',
		metavar='dataset',
		type=str,
		help='the path to dataset')
	my_parser.add_argument('-t',
		'--timer',
		action='store_true',
		help='Display time taken. Default false')
	my_parser.add_argument('-c',
		'--cost',
		action='store_true',
		help='Display cost graph, prediction error over training period. Default false')
	args = my_parser.parse_args()
	path = args.Dataset
	data = tools.read_csv(path)
	timer = args.timer
	cost = args.cost
	return data, timer, cost