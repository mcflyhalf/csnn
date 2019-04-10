#Implementation of neural network class

#Got tons of inspiration from https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import copy
import random
from csnn import get_logger

log = get_logger("csnn")

class NeuralNet:
	def __init__(self, layer_size):
		self.num_layers = pass
		self.layer_size = pass
		self.bias = list()
		self.weight = list()
		#For each layer (from layer 2 onwards), create a random bias that corresponds to each neuron.

		pass
		

		#Above 2 lines can also be expressed as:
		# self.bias = [np.random.randn(y, 1) for y in layer_size[1:]]

		#For each layer from layer 1 to layer n-1, create a (random) set of weights such that between any 2 layers l and l+1 in the network, there is a mweights matrix. The size of the weights matrix if layer l has x neurons and layer l+1 has y neurons should be l X y. This means that if layer l has 10 neurons and layer l+1 has 3 neurons, the weights matrix between these 2 will be a 10 X 3 matrix.
		
		pass

		#The whole of the above for loop can be condensed to:
		#self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


	def feedforward(self, inputs):
		'''A function which accepts an input vector and propagates it forward through the network and returns an output vector'''
		pass


	def train(self, training_data, epochs=30, mini_train_batch_size=-1, learn_rate=3.0, test_data=None):	#Train the network using stochastic gradient descent with what I think are decent default params
		'''Take the training data, split it into small batches and for each batch use that data to update the weights of the network using the update_weights fxn'''
		training_data = list(training_data)
		train_num = len(training_data)
		if mini_train_batch_size <1:
			mini_train_batch_size = train_num

		mini_train_batch_size = int(mini_train_batch_size)

		if(test_data):
			test_data = list(test_data)
			test_num = len(test_data)

		#What is the difference between training and test data??

		batch_strength = mini_train_batch_size/train_num
		for epoch in range(epochs):
			'''
			In each epoch:
			1. Shuffle the training data
			2.Create training batches then for each batch:
				i) update_weights()
			3. Test the performance against the test set (Implemented below)
			'''
			pass

			if test_data:
				hits = self.evaluate(test_data)
				print("Epoch {} : {} / {} >> {}% correct".format(epoch,hits,test_num, 100*(hits/test_num)))
			else:
				print("Epoch {} complete".format(epoch))	

	def update_weights(self,training_batch, batch_strength, learn_rate=3.0):
		'''Function that expects to get a part (or all of) the training data and use that to update the weights and biases of the NN. The batch strength is the ratio of the length of the training batch to that of the training data e.g. if there are 100 training data and a particular batch has 20 training samples, then batch strength is 20/100 = 0.2. Thus 0<batch_strength<=1'''

		#b for bias, w for weight
		#Copy the structure of self.bias and self.weight but fill it with all zeros.
		new_b = [np.zeros(b.shape) for b in self.bias]
		new_w = [np.zeros(w.shape) for w in self.weight]

		
		'''
		For each (input, target) pair in the training batch:
		1. Find the rate of change of all b and all w using backprop. Call this del_b and del_w
		2. Add del_b to the new_b variable above so that after all training, new_b has the sum of all del_b from the training run. Do the same for new_w
		3. Update the weights of the nn (self.weights) as weight-(learnrate*batchstrength*new_b). You will need a for loop to do this as self.weights is a list of matrices. Repeat for the biases as well
		'''

		pass
		

	def backprop(self, inpt, target):		#Backpropagation rule- Make a single pass forward in the NN, then propagate the error backwards. We don't use the feedforward function here because we need to keep track of the neuron activations

		current_activation = inpt
		activation = [inpt]	#Now contains the input to layer 1 (which is also the output of layer 1 because of the nature of the input layer), will eventually contain inputs to all layers

		z = list()		#Will store the inputs, z, to the neurons from each layer. z=w*inpt+b. Only layer 2 onwards have a z...

		#FORWARD PASS
		for b,w in zip(self.bias, self.weight):
			#calculate z for this layer, call it _z.
			#Append _z to the list of z
			#Get the current_activation functions for this layer as sigmoid(_z)
			#Append the current_activation to the list of activations
			pass

		log.debug("Size z[0]: {} \tSize z[1]: {}\t\tLen(Activation): {}".format(len(z[0]),len(z[1]),len(activation)))

		#BACKWARD PASS
		#Create a structure new_b and new_w that are exactly the same shape as self.weight and self.bias (use copy.deepcopy)
		#b for bias, w for weight
		new_b = copy.deepcopy(self.bias)
		new_w = copy.deepcopy(self.weight)

		#In the last layer:
		#del_b = 2*(output-target)*sigmoid'(last_z)
		#del_w = del_b* activation of the previous layer
		#Set the last element of new_b as del_b and the last element of new_w as del_w
		del_b = pass	#Rate of change of b for last layer wrt Cost function
		new_b[-1] = del_b
		new_w[-1] = pass		#Rate of change of w for last layer wrt Cost function
		log.debug("new_w shape: {}".format(new_w[-1].shape))
		new_w[-1] = new_w[-1].transpose()

		#TODO: Include logic to make this work for a nn with no hidden layers

		#Calculate the rate of change of w and b wrt cost function for each layer except the last which is done above...
		#In all other layers:
		#del_b = 2*(output-target)*sigmoid'(last_z)
		#del_w = del_b* activation of the previous layer
		#Set the last element of new_b as del_b and the last element of new_w as del_w. (Done earlier)
		for layer in range(2,self.num_layers):
			'''
			For each layer starting from the second-last to the first:
			1. Let _z be the z for that layer
			2. Get the sigmoid_prime of _z as sp
			3. Get del_b as self.weight[layer + 1] * del_b(from previous iteration) * sp
			4. new_b[layer] = this del_b
			5. new_w[layer] = del_b * activation[layer-1]

			'''
			pass

		return (new_b, new_w)		#Rates of change of b and w wrt Cost fxn should be same size and shape as self.bias and self.weight respectively

	def evaluate(self,test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result."""

		#storing results
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		#print(test_results)

		#checking how many are correct
		a = [int(x == y) for (x, y) in test_results]

		return sum(a)


def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))
