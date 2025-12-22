import numpy as np
import nnfs
from nnfs.datasets import spiral_data

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100, 3) # 100 feature sets of 3 classes, each feature set has 2 values x and y coordinates

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)     
        self.bias = np.zeros((1, n_neurons)) 


    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) #if inputs <=0 then 0 else input 


layer1 = Layer_Dense(2,5) 

layer1.forward(X)
activation1 = Activation_ReLU()
print(layer1.output)

activation1.forward(layer1.output)
print(activation1.output)
