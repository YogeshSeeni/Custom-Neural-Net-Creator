import numpy as np
from typing import List, Union, Optional
from dense import Dense
from activation_layer import ActivationLayer
from activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_prime
from loss_functions import mean_squared_error, mean_squared_error_derivative

class Model():
    def __init__(self) -> None:
        self.layers = []

    def add(self, layer: Union[Dense, ActivationLayer]) -> None:
        self.layers.append(layer)

    def predict(self, input: List) -> List:
        #Stores list of all outputs for each input 
        output = []

        for i in range(len(input)):
            #Keeps track of current output for the ith input (By Default the first is the input layer)
            singleOutput = input[i]

            #Forward propogate each layer using the output of the last layer
            for layer in self.layers:
                singleOutput = layer.forward_prop(singleOutput)
            
            #Add to the output list
            output.append(singleOutput)

        return output
    
    def print_error(self, epoch: int, error: float) -> None:
        print(f"Loss of Epoch #{epoch}: {error}")

    def fit(self, x: List, y: List, loss: Union[relu, sigmoid, tanh], loss_derivative: Union[relu_derivative, sigmoid_derivative, tanh_prime], epochs: Optional[int]=100, learning_rate: Optional[float]=0.01, verbosity: Optional[int]=1):
        x = np.array([[i] for i in x])
        y = np.array([[i] for i in y])

        for i in range(epochs):
            #Keeps track of error for current epoch for potential display at the end (depends on level of verbosity)
            total_error = 0

            for j in range(len(x)):
                #Keeps track of current output (starts with the input layer)
                output = x[j]

                #Forward propogate each layer using the output of the last layer
                for layer in self.layers:
                    output = layer.forward_prop(output)

                #Update the total error
                total_error += loss(y[j], output)

                #Backpropogation: Update weights and biases for each layer using derivative of error with respect to the output
                dE_dY = loss_derivative(y[j], output)
                #Loop backwards over layers as ∂E/∂X of the last layer is ∂E/∂Y of the layer before it
                for layer in reversed(self.layers):
                    dE_dY = layer.backward_prop(dE_dY, learning_rate)

            average_error = total_error / len(x)

            if verbosity == 0:
                continue
            elif verbosity == 1:
                if (i+1) % 100 == 0:
                    self.print_error(i+1, average_error)
            elif verbosity == 2:
                if (i+1) % 10 == 0:
                    self.print_error(i+1, average_error)
            else:
                self.print_error(i+1, average_error)