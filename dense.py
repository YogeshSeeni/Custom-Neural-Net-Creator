from baselayer import BaseLayer
import numpy

class Dense(BaseLayer):
    def __init__(self, input_shape, output_shape):
        # Randomly initialize weights for each connection between input and output (-0.5 to 0.5)
        self.weights = numpy.random.rand(input_shape, output_shape) - 0.5 

        #Initialize a 0 for each output neuron as a bias
        self.bias = numpy.zeros((1, output_shape))

    def forward_prop(self, x):
        #Save input for backpropagation
        self.input = x

        #Return output based on given input by doing the dot product between the input and weights plus the bias
        return numpy.dot(x, self.weights) + self.bias

    def backward_prop(self, dE_dY, learning_rate=0.01):
        #Calculate ∂E/∂W --> ∂E/∂W = ∂Y/∂W * ∂E/∂Y
        dE_dW = numpy.dot(self.input.T, dE_dY) #Note: T is transpose (flips shape of matrix)

        #Calculate ∂E/∂X --> ∂E/∂X = ∂E/∂Y * ∂Y/∂X
        dE_dX = numpy.dot(dE_dY, self.weights.T)

        dE_dB = dE_dY

        #Gradient Descent: Update weights and biases in the opposite direction of the gradient proportionately using learning rate
        self.bias -= dE_dB * learning_rate
        self.weights -= dE_dW * learning_rate

        #Return ∂E/∂X so it can be used as ∂E/∂Y for previous layer
        return dE_dX
