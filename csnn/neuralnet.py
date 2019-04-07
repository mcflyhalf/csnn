#Implementation of neural network class

#Got tons of inspiration from https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import copy

class NeuralNet:
	def __init__(self, layer_size):
		self.num_layers = len(layer_size)
		self.layer_size = layer_size
		self.bias = list()
		self.weight = list()
		#For each layer (from layer 2 onwards), create a random bias that corresponds to each neuron. Biases are normalised from -1 to 1 
		for size in layer_size[1:]:
			self.bias.append(np.random.randn(size,1))

		#Above 2 lines can also be expressed as:
		# self.biase = [np.random.randn(y, 1) for y in layer_size[1:]]

		for size1, size2 in zip(layer_size[1:],layer_size[:-1]):
			self.weight.append(np.random.randn(size2,size1))

		#The whole of the above for loop can be condensed to:
		#self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def feedforward(self, inputs):
		for b,w in zip(self.bias, self.weight):		#b is a single bias for a neuron n in layer L and w is a list of weights for each edge entering n from layer L-1
			inputs = np.dot(w,inputs) + b
			inputs=sigmoid(inputs)

		return inputs		#Despite its deceptive name, inputs now contains outputs. This nomenclature was useful for conciseness of the code

	def train(self):	#Train the network using stochastic gradient descent
		pass


	def backprop(self, input, expected_output):		#Backpropagation rule- Make a single pass forward in the NN, then ...

		current_activation = input
		activation = [input]	#Now contains the input to layer 1, will eventually contain inputs to all layers

		z = list()		#Will store the outputs, z, of the neurons from each layer. z=w*input+b

		#Forward pass
		for b,w in zip(self.biases, self.weights):
			#calculate zfor this layer, call it _z
			_z = np.dot(w,current_activation)+b
			z.append(_z)
			current_activation= sigmoid(_z)
			activation.append(current_activation)

		#Backward pass

		#b for bias, w for weight
		new_b = copy.deepcopy(self.bias)
		new_w = copy.deepcopy(self.weight)

		del_b = (activation[-1] - expected_output) * sigmoid_prime(z[-1])	#Rem -(y-y') = y'-y
		new_b[-1] = del_b
		new_w[-1] = np.dot(del_b,activation[-2].transpose)

		for layer in range(2,self.num_layers):
			_z = z[-layer]
			sp = sigmoid_prime(_z)
			#What is happening here? Can't understand!










