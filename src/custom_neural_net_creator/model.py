import numpy as np
from typing import List, Union, Optional
from custom_neural_net_creator.dense import Dense
from custom_neural_net_creator.activation_layer import ActivationLayer
from custom_neural_net_creator.loss_functions import mean_squared_error, mean_squared_error_derivative

class Model():
    def __init__(self) -> None:
        """Initialize a neural network model.

        This class represents a neural network model with a list of layers.
        """
        self.layers = []

    def add(self, layer: Union[Dense, ActivationLayer]) -> None:
        """Adds a layer to neural network model.

        This method appends a layer, which can be either a Dense layer or Activation Layer,
        to the list of layers in the neural network model.
        
        Args:
            layer (Union[Dense, ActivationLayer]): Layer to Be Added to Neural Network
        """
        self.layers.append(layer)

    def predict(self, input: List) -> List:
        """Predicts the output for a given input using the neural network model.
        
        This method takes a list of input data and computes predictions using the neural network model. 
        It iterates through each input, performing forward propagation through the network's layers 
        and collecting the output. The predictions are stored in a list and returned.

        Args:
            input (List): Input data to make predictions on.

        Returns:
            List: List of predictions for the input data.
        """
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
        """Prints the error for a given epoch.

        This method prints the loss (error) for a specific training epoch. It is typically used for monitoring
        the training progress. The `epoch` parameter specifies the current epoch number, and `error` is the
        calculated loss for that epoch.

        Args:
            epoch (int): The epoch number.
            error (float): The error value for the epoch.
        """
        print(f"Loss of Epoch #{epoch}: {error}")

    def fit(self, x: List, y: List, loss: mean_squared_error, loss_derivative: mean_squared_error_derivative, epochs: Optional[int]=100, learning_rate: Optional[float]=0.01, verbosity: Optional[int]=1):
        """Trains the neural network model on the given data.

        This method trains the neural network model using the specified input data `x`, target data `y`, 
        loss function `loss`, and its derivative `loss_derivative`. It performs a number of training epochs
        specified by the `epochs` parameter, updating the model's weights and biases through backpropagation
        with gradient descent.

        The `learning_rate` parameter controls the step size during weight updates. The `verbosity` parameter
        determines the level of output during training, with options to print loss values at different intervals.

        Args:
            x (List): Input data for training.
            y (List): Target data for training.
            loss (mean_squared_error): The loss function used for training.
            loss_derivative (mean_squared_error_derivative): The derivative of the loss function.
            epochs (Optional[int], optional): Number of training epochs. Defaults to 100.
            learning_rate (Optional[float], optional): Learning rate for gradient descent. Defaults to 0.01.
            verbosity (Optional[int], optional): Verbosity level for printing training progress. 0 - No output, 1 - Print every 100 epochs, 2 - Print every 10 epochs. Defaults to 1.
        """
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