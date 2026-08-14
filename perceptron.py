import numpy as np

class Perceptron:
    def __init__(self, weights,bias):
        self.weights=weights
        self.bias=bias

    def pre_activation(self,x):
        return np.dot(self.weights,x)+self.bias

    def activation(self,z):
        return max(0,z)