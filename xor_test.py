import numpy as np

from model import Model
from dense import Dense
from activation_layer import ActivationLayer
from activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_prime
from loss_functions import mean_squared_error, mean_squared_error_derivative

x = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y = np.array([[[0]], [[1]], [[1]], [[0]]])

model = Model()

model.add(Dense(2, 10)) #Input takes in two inputs
model.add(ActivationLayer(relu, relu_derivative))
model.add(Dense(10, 10))
model.add(ActivationLayer(relu, relu_derivative))
model.add(Dense(10,1))
model.add(ActivationLayer(sigmoid, sigmoid_derivative))

model.fit(x,y,mean_squared_error,mean_squared_error_derivative,epochs=1000,learning_rate=0.1,verbosity=3)

predictions = model.predict(x[0:3])

print("Predicted: ")
print(predictions)

print("Actual:")
print(y[0:3])
