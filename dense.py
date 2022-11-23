from baselayer import BaseLayer
import numpy

class Dense(BaseLayer):
    def __init__(self, input_shape, output_shape):
        # Randomly initialize weights for each connection between input and output
        self.weights = numpy.random.rand(input_shape, output_shape) - 0.5 

        #Initialize Biases as 0
        self.bias = numpy.zeros((1, output_shape))


Dense(2,3)
