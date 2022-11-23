import numpy

def mean_squared_error(actual, prediction):
    return numpy.mean(numpy.power(actual - prediction, 2))

def mean_squared_error_derivative(actual, prediction):
    return 2/actual.size*(prediction-actual)