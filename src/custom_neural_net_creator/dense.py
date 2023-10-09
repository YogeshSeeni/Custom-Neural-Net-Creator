import numpy
import numpy.typing as npt

class Dense():
    def __init__(self, input_shape: int, output_shape: int) -> None:
        """Intialize a Dense layer.

        This constructor initializes a Dense layer with randomly initialized weights and biases.
        The `input_shape` parameter specifies the number of input features, and the `output_shape`
        parameter specifies the number of output neurons. Weights are initialized within the range
        (-0.5 to 0.5), and biases are initialized to zero.

        Args:
            input_shape (int): The number of input neurons.
            output_shape (int): The number of output neurons.
        """
        # Randomly initialize weights for each connection between input and output (-0.5 to 0.5)
        self.weights = numpy.random.rand(input_shape, output_shape) - 0.5 

        #Initialize a 0 for each output neuron as a bias
        self.bias = numpy.zeros((1, output_shape))

    def forward_prop(self, x: npt.NDArray) -> npt.NDArray:
        """Perform forward propagation through the Dense layer.

        This method computes the forward propagation of input data through the Dense layer. 
        The output is calculated by performing a dot product between the input and the layer's weights, 
        adding the bias, and returning the result. It saves
        the input `x` for later use in backpropagation. 

        Args:
            x (npt.NDArray): The input data of the Dense layer.

        Returns:
            npt.NDArray: The output of the Dense layer.
        """
        #Save input for backpropagation
        self.x = x

        #Return output based on given input by doing the dot product between the input and weights plus the bias
        self.y = numpy.dot(self.x, self.weights) + self.bias
        return self.y

    def backward_prop(self, dE_dY: npt.NDArray, learning_rate: float) -> npt.NDArray:
        """Perform backward propagation through the Dense layer and update weights and biases.

        This method performs backward propagation through the Dense layer. It calculates the gradients
        of the error with respect to the layer's weights, biases, and input. The gradients are then used
        to update the weights and biases using gradient descent. The method returns the gradient of the
        error with respect to the layer's input, to be used for backpropagation in previous layers.

        Args:
            dE_dY (npt.NDArray): The gradient of the error function.
            learning_rate (float): The learning rate for perorming gradient descent.

        Returns:
            npt.NDArray: The gradient of the error with respect to current layer's input.
        """
        #Calculate ∂E/∂W --> ∂E/∂W = ∂Y/∂W * ∂E/∂Y
        dE_dW = numpy.dot(self.x.T, dE_dY) #Note: T is transpose (flips shape of matrix)

        #Calculate ∂E/∂X --> ∂E/∂X = ∂E/∂Y * ∂Y/∂X
        dE_dX = numpy.dot(dE_dY, self.weights.T)

        dE_dB = dE_dY

        #Gradient Descent: Update weights and biases in the opposite direction of the gradient proportionately using learning rate
        self.bias -= dE_dB * learning_rate
        self.weights -= dE_dW * learning_rate

        #Return ∂E/∂X so it can be used as ∂E/∂Y for previous layer
        return dE_dX
