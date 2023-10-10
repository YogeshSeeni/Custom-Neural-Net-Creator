from custom_neural_net_creator.activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_prime
from typing import Union
import numpy.typing as npt

class ActivationLayer():
    def __init__(self, activation_function: Union[relu, sigmoid, tanh], activation_derivative: Union[relu_derivative, sigmoid_derivative, tanh_prime]):
        """Iniitalize an Activation Layer with a specified activation function.

        Args:
            activation_function (Union[relu, sigmoid, tanh]): The activation function to be added.
            activation_derivative (Union[relu_derivative, sigmoid_derivative, tanh_prime]): The activation function's derivative to be added.
        """
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward_prop(self, x: npt.NDArray) -> npt.NDArray:
        """Perform forward propagation through the Activation Layer.

        This method computes the forward propagation of input data through the Activation Layer. It saves
        the input `x` for later use in backpropagation and passes the input through the specified activation
        function. The result is returned as the output of the layer.

        Args:
            x (npt.NDArray): The input data for the activation layer.

        Returns:
            npt.NDArray: The output of the activation layer.
        """
        self.x = x
        #pass input into activation function
        self.y = self.activation_function(self.x)
        return self.y
    
    def backward_prop(self, dE_dY: npt.NDArray, learning_rate: float) -> npt.NDArray:
        """This method performs backward propagation through the Activation Layer. It computes the gradient
        of the error with respect to the layer's input by multiplying the activation derivative with
        ∂E/∂Y. Since there are no parameters to optimize in this layer, it directly returns the gradient
        of the error with respect to the input, which can be used for backpropagation in previous layers.

        Args:
            dE_dY (npt.NDArray): The gradient of the error function.
            learning_rate (float): The learning rate for perorming gradient descent.

        Returns:
            npt.NDArray: The gradient of the error with respect to current layer's input.
        """
        #Return multiplication of activation derivative and ∂E/∂Y as there is not parameters to optimize in this layer
        return self.activation_derivative(self.x) * dE_dY
        