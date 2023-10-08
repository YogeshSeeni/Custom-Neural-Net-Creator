import numpy
import numpy.typing as npt

def mean_squared_error(actual: npt.NDArray, prediction: npt.NDArray) -> npt.NDArray:
    return numpy.mean(numpy.power(actual - prediction, 2))

def mean_squared_error_derivative(actual: npt.NDArray, prediction: npt.NDArray) -> npt.NDArray:
    return 2/actual.size*(prediction-actual)