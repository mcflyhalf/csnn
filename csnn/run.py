import loader
import neuralnet


training_data, validation_data, test_data = loader.load_data_wrapper()
training_data = list(training_data)

nn = neuralnet.NeuralNet([784, 30, 10])
#making network
nn.train(training_data, 30, 10, 3.0, test_data=test_data)
#running network