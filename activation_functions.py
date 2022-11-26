import numpy

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return numpy.maximum(0,x)

def relu_derivative(x):
    copyX = x
    copyX[copyX < 0] = 0
    copyX[copyX > 0] = 1
    return x 

def tanh(x):
    return numpy.tanh(x);

def tanh_prime(x):
    return 1-numpy.tanh(x)**2;