# Custom Neural Net Creator

## Overview

[Custom-Neural-Net-Creator](https://pypi.org/project/custom-neural-net-creator/) is an easy, to use tool that allows developers and machine learning enthusiasts to create and deploy their own custom deep neural networks for various purposes. This innovative tool makes the power of learning more accessible, than before.

Neural networks are advanced machine learning algorithms that can detect complex patterns in large amounts of 
data and make predictions based on the information they have been trained on. 
These systems are designed to mimic the structure of the human brain, with interconnected layers of 
“neurons” that transmit information to each other. 
An artificial neural network typically consists of an input layer, 
one or more hidden layers, and an output layer, each with a specific number of neurons connected to 
preceding and following layers. Deep neural networks are a subtype of artificial neural networks 
that can learn from large datasets and make predictions based on them.

To view more information visit the [documentation](https://custom-neural-net-creator.readthedocs.io/en/latest/).

## Example

This is an example of how to use this module on a the classic XOR Problem

```python
import numpy as np

from custom_neural_net_creator.model import Model
from custom_neural_net_creator.dense import Dense
from custom_neural_net_creator.activation_layer import ActivationLayer
from custom_neural_net_creator.activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_prime
from custom_neural_net_creator.loss_functions import mean_squared_error, mean_squared_error_derivative

#Input data for XOR
x = [[0,0], [0,1], [1,0], [1,1]]
y = [[0], [1], [1], [0]]

model = Model()

model.add(Dense(2, 10)) #Input takes in two inputs
model.add(ActivationLayer(relu, relu_derivative)) #First hidden layer has 10 neurons and uses RELU
model.add(Dense(10, 10))
model.add(ActivationLayer(relu, relu_derivative)) #Second hidden layer has 10 neurons and uses RELU
model.add(Dense(10,1))
model.add(ActivationLayer(sigmoid, sigmoid_derivative)) #Output layer is one neuron with Sigmoid as activation

#Train on training data
model.fit(x,y,mean_squared_error,mean_squared_error_derivative,epochs=1000,learning_rate=0.1,verbosity=3)
#Loss of Epoch #1000: 0.0002757698731393589

#Test model
predictions = model.predict(x[0:3])

print("Predicted: ")
print(predictions) #Predicted: [array([[0.02610931]]), array([[0.98778214]]), array([[0.9873547]])]

print("Actual:")
print(y[0:3])
# Actual:
# [[[0]]

# [[1]]

# [[1]]]
```