
# Multilayer-Perceptron

A deep neural network that predicts whether a cancer is malignant or benign. <br>
Built from scratch, using python numpy. 

Based on the Wisconsin breast cancer diagnosis dataset.

### Requirements:
* the network should consist of at least 4 fully-connected (dense) layers 
* the following must be implemented from scratch (using only numpy):
	-  gradient descent
	-  the softmax function
	-  feedforward
	-  backpropagation
	-  binary cross-entropy error
* the final loss should be below 0.08

#### Final Score: 125/100

## Getting Started

First clone this repo. <br>
```git clone https://github.com/anyashuka/Multilayer-Perceptron.git; cd Multilayer-Perceptron```

Download dependencies. <br>
```pip3 install -r requirements.txt```

Then simply run main.py with your choice of flags.

To visualize the data: <br>
```python3 main.py data.csv -v```

To train the model: <br>
```python3 main.py data.csv -t -s {optional -b -e -q}```

To load a trained model and test: <br>
```python3 main.py data.csv -p model.json```

### Usage

![usage](https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/usage.png)

### Flags

**-v (--visualize_data)** <br>
Displays pair plots, heat maps, and strip plots chosen from an array of 13 features. <br>
Built using scikit-learn. Here is the original dataset visualized:

![dataset](https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/dataset.png)

Here is a simple example with radius displayed as a strip plot, showing that radius_mean can potentially help differentiate between malignant and benign tumors.

<img src="https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/radius_strip_plot.png" width="500">

**-t (--train)** <br>
Outputs the loss at every epoch, allowing us to verify that the loss is in fact going down. 

![streaming loss](https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/streaming_loss.gif)

Once the model is finished training, we get a visual of the learning curve. This helps us verify that our loss has reached a global minimum. 

<img src="https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/learning_curve.png" width="500">

**-b (--mini_batch)** <br>
Trains the neural network on mini-batches of size 32. <br> 
The advantage here is it introduces enough noise in each gradient update to avoid getting trapped in local minima, while allowing a fast convergence (as opposed to batch or SGD).

**-e evaluation** <br>
Outputs metrics which give us a deeper look into the model's performance.  

![evaluation metrics](https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/evaluation_metrics.png)

## Concepts

**Data Processing**
Have you cleaned your data? Typically, this means:
- getting rid of useless data (like patient ids)
- cleaning your dataset of erroneous or invalid data points (NaNs)
- standardizing your data: centering all data points around the mean (zero) with a unit standard deviation (min, max)

**Matrix Multiplication**
Click [here](http://matrixmultiplication.xyz/) for a handy little refresher. 

![Matrix Multiplication](https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/matrix_multiplication.gif)

**Feedforward**
This means the data flows from the input layer to the output layer.

**Backpropagation** <br>
Backpropagation is an application [the Chain rule](https://www.youtube.com/watch?v=tIeHLnjs5U8&t=38s) to find the derivatives of cost with respect to any variable in the nested equation. This simple technique allows us to precisely pinpoint the exact impact each variable has on the total output.

**The Chain Rule** <br>
This helps us identify how much each weight contributes to our overall error, as well as the direction to update each weight to reduce our error.

**Gradient Descent** <br>
We use gradient descent to update the parameters (weights) of our model. The gradient (or derivative) of the loss function tells us the direction we need to adjust our weights in order to achieve the lowest loss (i.e. the smallest number of bad classifications). 
Imagine you're at the top of a hill gazing down at a valley. Gradient descent will help you find the quickest path to the bottom of the valley. 

**Activation Function** <br>
This is what allows neural networks to {}.
An Activation function decides whether a neuron should be activated or not by calculating the weighted sum of its inputs and adding bias to it. 
Activation functions are non-linear, as seen below. 

<img src="https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/sigmoid.png" width="300"> <img src="https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/softmax.png" width="300"> <br>
<img src="https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/ReLU.png" width="300"> <img src=https://github.com/anyashuka/Multilayer-Perceptron/blob/master/img/leaky_ReLU.png width="300">

Choosing an activation function depends on your application and on the architecture of your neural network. I implemented the above activations from scratch, using numpy. My neural network uses sigmoid in the hidden layers and softmax in the output layer. Unfortunately, I found that both ReLU and leakyReLU units (in the hidden layers) would die off during training. (see "Vanishing/Exploding Gradients below)

**Loss Function**<br>
The loss function outputs a value that represents how well (or badly) our model is doing. <br>
High loss means our classifier’s predictions are mostly wrong, so our model is bad.  <br>
Low loss means our classifier’s predictions are mostly correct, so our model is good! <br>
We use the loss function to evaluate the “goodness” of our model’s weights. <br>

**Learning Rate**
The size of step we take when applying Gradient Descent to update the parameters (weights and biases) of our model. 

**Vanishing/Exploding Gradients**<br>
Vanishing gradients is a problem where the gradient will decrease exponentially as we propagate through the model until it eventually vanishes, making it impossible to update your weights and continue training your model.
If the local gradient is very small, it will effectively "kill" the gradient and almost no signal will flow through the neuron to its weights and recursively to its data.

Exploding gradients is a problem where large error gradients accumulate and result in very large updates to neural network model weights during training. This has the effect of your model being unstable and unable to learn from your training data.

**Optimizations**
- He weight initialization: the weights are still random but differ in range depending on the size of the previous layer of neurons. This provides a controlled initialization hence the faster and more efficient gradient descent.

- I observed the model had 98% accuracy when the data was standardized, but hovered around ~60% accuracy when it was normalized. 
