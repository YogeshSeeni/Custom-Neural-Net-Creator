import numpy
import numpy.typing as npt

def mean_squared_error(actual: npt.NDArray, prediction: npt.NDArray) -> float:
    """Calculate the mean squared error between acuatl and predicted values.

    This function computes the mean squared error (MSE) between the actual and predicted values.
    It takes two NumPy arrays, `actual` and `prediction`, and calculates the squared difference 
    between corresponding elements, then returns the mean of those squared differences. Lower values 
    indicate better performance during training.

    Args:
        actual (npt.NDArray): The actual values.
        prediction (npt.NDArray): The predicted values.

    Returns:
        float: The mean squared error between actual and predicted values.
    """
    return numpy.mean(numpy.power(actual - prediction, 2))

def mean_squared_error_derivative(actual: npt.NDArray, prediction: npt.NDArray) -> float:
    """Calculate the derivative of the mean squared error with respect to predicted values.

    This function computes the mean squared error (MSE) between the actual and predicted values.
    It takes two NumPy arrays, `actual` and `prediction`, and calculates the squared difference 
    between corresponding elements, then returns the mean of those squared differences. MSE is a
    common metric used to measure the accuracy of regression models. Lower values indicate better 
    performance during training.

    Args:
        actual (npt.NDArray): The actual values.
        prediction (npt.NDArray): The predicted values.

    Returns:
        float: The derivative of the mean squared error with respect to predicted values.
    """
    return 2/actual.size*(prediction-actual)