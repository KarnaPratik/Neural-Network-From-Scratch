import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): # Make a random set of weights
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        #Multiply by 0.1 so that all values are in between  0 and 1        
        #print(np.random.randn(4, 3)) Used to check the intial set of values for self.weight

        self.bias = np.zeros((1, n_neurons)) 
        #May also make a dead network if all values give an output of 0, so make sure to initialize bias to sth else in case that happens.

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


layer1 = Layer_Dense(4,5) #4 inputs in one list and 5 is the no of neurons for eg.
layer2 = Layer_Dense(5,2) #2 is a random no of neurons eg again.
# since output of layer1 is the input for layer2 the shape should be same, so layer 2 is of the shape 5 since 5 neurons in the first network. layer1 shape is 4 btw.

layer1.forward(X)
# print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)
