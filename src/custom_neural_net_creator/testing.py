from model import Model
from dense import Dense
from activation_layer import ActivationLayer
from activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_prime
from loss_functions import mean_squared_error, mean_squared_error_derivative

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
model.fit(x,y,mean_squared_error,mean_squared_error_derivative)
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