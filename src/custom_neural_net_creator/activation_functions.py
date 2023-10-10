import numpy
import numpy.typing as npt

def sigmoid(x: npt.NDArray) -> npt.NDArray:
    """Calculate the sigmoid activation function.

    This function computes the sigmoid activation function for the given input data `x`. The sigmoid
    function maps input values to the range (0, 1), making it suitable for binary classification problems.
    It returns the result of the sigmoid activation applied to each element of the input array.

    Args:
        x (npt.NDArray): The input data.

    Returns:
        npt.NDArray: The output of sigmoid activation function.
    """
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x: npt.NDArray) -> npt.NDArray:
    """Calculate the derivative of the sigmoid activation function.

    This function computes the derivative of the sigmoid activation function for the given input data `x`.
    The sigmoid derivative is used in backpropagation to calculate gradients during neural network training.
    It returns the result of the sigmoid derivative applied to each element of the input array.

    Args:
        x (npt.NDArray): The input data.

    Returns:
        npt.NDArray: The derivative of the sigmoid activation function.
    """
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x: npt.NDArray) -> npt.NDArray:
    """Calculate the ReLU (Rectified Linear Unit) activation function.

    This function computes the ReLU activation function for the given input data `x`. The ReLU function
    is commonly used in neural networks and returns the maximum of 0 and the input value for each element
    of the input array. It effectively introduces non-linearity to the network.

    Args:
        x (npt.NDArray): The input data.

    Returns:
        npt.NDArray: The output of the ReLU activation function.
    """
    return numpy.maximum(0,x)

def relu_derivative(x: npt.NDArray) -> npt.NDArray:
    """Calculate the derivative of the ReLU (Rectified Linear Unit) activation function.

    This function computes the derivative of the ReLU activation function for the given input data `x`.
    The ReLU derivative is used in backpropagation to calculate gradients during neural network training.
    It returns the result of the ReLU derivative applied to each element of the input array.

    Args:
        x (npt.NDArray): The input data.

    Returns:
        npt.NDArray: The derivative of the ReLU activation function.
    """
    copyX = x
    copyX[copyX < 0] = 0
    copyX[copyX > 0] = 1
    return x 

def tanh(x: npt.NDArray) -> npt.NDArray:
    """Calculate the hyperbolic tangent (tanh) activation function.

    This function computes the hyperbolic tangent (tanh) activation function for the given input data `x`.
    The tanh function maps input values to the range (-1, 1) and is used in neural networks to introduce
    non-linearity. It returns the result of the tanh activation applied to each element of the input array.

    Args:
        x (npt.NDArray): The input data.

    Returns:
        npt.NDArray: The output of the tanh activation function.
    """
    return numpy.tanh(x);

def tanh_prime(x: npt.NDArray) -> npt.NDArray:
    """Calculate the derivative of the hyperbolic tangent (tanh) activation function.

    This function computes the derivative of the hyperbolic tangent (tanh) activation function for the given
    input data `x`. The tanh derivative is used in backpropagation to calculate gradients during neural
    network training. It returns the result of the tanh derivative applied to each element of the input array.

    Args:
        x (npt.NDArray): The input data.

    Returns:
        npt.NDArray: The derivative of the tanh activation function.
    """
    return 1-numpy.tanh(x)**2;