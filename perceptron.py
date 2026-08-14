import numpy as np

class Perceptron:
    def __init__(self, weights,bias):
        self.weights=None
        self.bias=None

    def pre_activation(self,x):
        return np.dot(self.weights,x)+self.bias