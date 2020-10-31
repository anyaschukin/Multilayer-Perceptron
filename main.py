import sys
import json
from tools import get_args
from preprocess import preprocess
from visualize import visualize
from neural_network import *

def main():
	# try:
	args = get_args()

	data = preprocess(args)

	if args.visualize_data:
		visualize(data)
		sys.exit(1) 

	train_set, test_set = split(data)
	
	num_examples = train_set.shape[0]
	num_features = train_set.shape[1] - 1
	if args.mini_batch:
		batch_size = 32		# or 64
		epochs = 1500
	else:
		batch_size = num_examples
		epochs = 30000

	nn = NeuralNetwork(num_features, batch_size, epochs)

	if args.train:
		nn.train(data, train_set, test_set, num_examples, args.quiet)

		if args.evaluation:
			y_pred = probability_to_class( nn.output.T)
			get_validation_metrics(y_pred[:, 0],  nn.y.T[:, 0])

		# mini-batch learning is noisy, so we don't plot it 
		if not args.mini_batch:
			plot_learning(nn.train_losses, nn.test_losses)

		# save network params
		if args.save_model:
			W1, W2, W3, W4 =  nn.weights1.tolist(),  nn.weights2.tolist(),  nn.weights3.tolist(),  nn.weights4.tolist()
			B1, B2, B3, B4 =  nn.bias1.tolist(),  nn.bias2.tolist(),  nn.bias3.tolist(),  nn.bias4.tolist()
			model = dict(weights1=W1, weights2=W2, weights3=W3, weights4=W4, bias1=B1, bias2=B2, bias3=B3, bias4=B4)
			with open("neural_network.json", "w") as f:
				json.dump(model, f, separators=(',', ':'), indent=4)

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