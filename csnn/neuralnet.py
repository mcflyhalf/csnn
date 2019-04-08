#Implementation of neural network class

#Got tons of inspiration from https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6

import numpy as np
import copy
import random
from csnn import get_logger

log = get_logger("csnn")

class NeuralNet:
	def __init__(self, layer_size):
		self.num_layers = len(layer_size)
		self.layer_size = layer_size
		self.bias = list()
		self.weight = list()
		#For each layer (from layer 2 onwards), create a random bias that corresponds to each neuron.
		for size in self.layer_size[1:]:
			self.bias.append(np.random.randn(size,1))

		#Above 2 lines can also be expressed as:
		# self.bias = [np.random.randn(y, 1) for y in layer_size[1:]]

		#For each layer from layer 1 to layer n-1, create a (random) set of weights such that between any 2 layers l and l+1 in the network, there is a mweights matrix. The size of the weights matrix if layer l has x neurons and layer l+1 has y neurons should be l X y. This means that if layer l has 10 neurons and layer l+1 has 3 neurons, the weights matrix between these 2 will be a 10 X 3 matrix.
		for size1, size2 in zip(layer_size[1:],layer_size[:-1]):
			self.weight.append(np.random.randn(size2,size1))

		#The whole of the above for loop can be condensed to:
		#self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


	def feedforward(self, inputs):
		'''A function which accepts an input vector and propagates it forward through the network and returns an output vector'''
		for b,w in zip(self.bias, self.weight):		#b is a single bias for a neuron n in layer L and w is a list of weights for each edge entering n from layer L-1
			inputs = np.dot(w.transpose(),inputs) + b 	#May be inputs = np.dot(w,inputs) + b 
			inputs=sigmoid(inputs)

		return inputs		#Despite its deceptive name, inputs now contains outputs. This nomenclature was useful for conciseness of the code

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

		batch_strength = mini_train_batch_size/train_num
		for epoch in range(epochs):
			random.shuffle(training_data)
			for k in range(0,train_num, mini_train_batch_size):
				mini_train_batch = training_data[k:k+mini_train_batch_size]
				self.update_weights(mini_train_batch, learn_rate, batch_strength)

			if test_data:
				print("Epoch {} : {} / {} >> {} % correct".format(epoch,self.evaluate(test_data),test_num, ));
			else:
				print("Epoch {} complete".format(epoch))	

	def update_weights(self,training_batch, batch_strength, learn_rate=3.0):
		'''Function that expects to get a part (or all of) the training data and use that to update the weights and biases of the NN. The batch strength is the ratio of the length of the training batch to that of the training data e.g. if there are 100 training data and a particular batch has 20 training samples, then batch strength is 20/100 = 0.2. Thus 0<batch_strength<=1'''

		#b for bias, w for weight
		#Copy the structure of self.bias and self.weight but fill it with all zeros. Could also use:
		#new_b = copy.deepcopy(self.bias) 
		#and a similar expression for the weights
		new_b = [np.zeros(b.shape) for b in self.bias]
		new_w = [np.zeros(w.shape) for w in self.weight]

		
		'''
		For each (input, target) pair in the training batch:
		1. Find the rate of change of all b and all w using backprop. Call this del_b and del_w
		2. Add del_b to the new_b variable above so that after all training, new_b has the sum of all del_b from the training run. Do the same for new_w
		3. Update the weights of the nn (self.weights) as weight-(learnrate*batchstrength*new_b). You will need a for loop to do this as self.weights is a list of matrices. Repeat for the biases as well
		'''
		
		for inpt, target in training_batch:
			delta_new_b,delta_new_w = self.backprop(inpt,target)

			new_b = [nb+dnb for nb, dnb in zip(new_b, delta_new_b)]
			new_w = [nw+dnw for nw, dnw in zip(new_w, delta_new_w)]

			self.weight = [w-((learn_rate*batch_strength)*nw) for w, nw in zip(self.weight, new_w)]
			self.bias = [b-((learn_rate*batch_strength)*nb) for b, nb in zip(self.bias, new_b)]


	def backprop(self, inpt, target):		#Backpropagation rule- Make a single pass forward in the NN, then propagate the error backwards. We don't use the feedforward function here because we need to keep track of the neuron activations

		current_activation = inpt
		activation = [inpt]	#Now contains the input to layer 1 (which is also the output of layer 1 because of the nature of the input layer), will eventually contain inputs to all layers

		z = list()		#Will store the inputs, z, to the neurons from each layer. z=w*inpt+b. Only layer 2 onwards have a z...

		#Forward pass
		for b,w in zip(self.bias, self.weight):
			#calculate z for this layer, call it _z
			_z = np.dot(w.transpose(),current_activation)+b
			z.append(_z)
			current_activation= sigmoid(_z)
			activation.append(current_activation)

		log.debug("Size z[0]: {} \tSize z[1]: {}\t\tLen(Activation): {}".format(len(z[0]),len(z[1]),len(activation)))

		#Backward pass

		#b for bias, w for weight
		new_b = copy.deepcopy(self.bias)
		new_w = copy.deepcopy(self.weight)

		del_b = (activation[-1] - target) * sigmoid_prime(z[-1])	#Rate of change of b for last layer wrt Cost function
		new_b[-1] = del_b
		new_w[-1] = np.dot(del_b,activation[-2].transpose())		#Rate of change of w for last layer wrt Cost function
		log.debug("new_w shape: {}".format(new_w[-1].shape))
		new_w[-1] = new_w[-1].transpose()

		#TODO: Include logic to make this work for a nn with no hidden layers

		#Need to understand this bit. Dont quite get how the maths translates to this. Calculate the rate of change of w and b wrt cost function for each layer except the last which is done above...
		for layer in range(2,self.num_layers):
			_z = z[-layer]
			sp = sigmoid_prime(_z)
			del_b = np.dot(self.weight[-layer+1], del_b) * sp
			new_b[-layer] = del_b
			new_w[-layer] = np.dot(del_b, activation[-layer-1].transpose())
			new_w[-layer] = new_w[-layer].transpose()

		return (new_b, new_w)		#Rates of change of b and w wrt Cost fxn should be sam size and shape as self.bias and self.weight respectively

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
