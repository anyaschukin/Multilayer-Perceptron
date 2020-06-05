import pandas as pd
import numpy as np
import tools as tools
import matplotlib.pyplot as plt
import preprocess as prep

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

def softmax_prime(z):
  return softmax(z) * (1-softmax(z))

def ReLU(Z):
    return np.maximum(0, Z)

def ReLU_prime(Z):
    return 1 * (Z > 0)

def leaky_ReLU(z, alpha = 0.01):
	return np.where(z >= 0, z, z * alpha)
    # return max(alpha * z, z)

def leaky_ReLU_prime(z, alpha = 0.01):
    return np.where(z >= 0, 1, alpha)
	# return 1 if x > 0 else alpha

def test(activation=softmax):
    x = np.array([-10, -9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9, 10])
    y = activation(x)
    title = str(activation).split(" ")[1]
    plt.suptitle(title)
    plt.plot(x, y)
    plt.show()

def main():
    test()
    test(sigmoid)
    test(ReLU)
    test(leaky_ReLU)

if __name__ == '__main__':
    main()




# 4e003
# 4000

# 4e-003
# 0.004

#10e-6
#0.00001