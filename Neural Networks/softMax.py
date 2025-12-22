import numpy as np
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(0)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)     
        self.bias = np.zeros((1, n_neurons)) 


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) #if inputs <=0 then 0 else input 

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims=True))
#axis 1 means max of the values in a single row and keepdims keeps the shape. axis 0 means a column
        probabilites = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probabilites


X, y= spiral_data(samples= 100, classes= 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])