import numpy
import numpy.typing as npt

def sigmoid(x: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x: npt.NDArray) -> npt.NDArray:
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x: npt.NDArray) -> npt.NDArray:
    return numpy.maximum(0,x)

def relu_derivative(x: npt.NDArray) -> npt.NDArray:
    copyX = x
    copyX[copyX < 0] = 0
    copyX[copyX > 0] = 1
    return x 

def tanh(x: npt.NDArray) -> npt.NDArray:
    return numpy.tanh(x);

def tanh_prime(x: npt.NDArray) -> npt.NDArray:
    return 1-numpy.tanh(x)**2;