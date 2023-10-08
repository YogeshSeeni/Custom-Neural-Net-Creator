from activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_prime
from typing import Union
import numpy.typing as npt

class ActivationLayer():
    def __init__(self, activation_function: Union[relu, sigmoid, tanh], activation_derivative: Union[relu_derivative, sigmoid_derivative, tanh_prime]) -> npt.NDArray:
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward_prop(self, x: npt.NDArray) -> npt.NDArray:
        self.x = x
        #pass input into activation function
        self.y = self.activation_function(self.x)
        return self.y
    
    def backward_prop(self, dE_dY, learning_rate):
        #Return multiplication of activation derivative and ∂E/∂Y as there is not parameters to optimize in this layer
        return self.activation_derivative(self.x) * dE_dY
        