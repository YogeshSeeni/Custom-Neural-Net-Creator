Custom Neural Net Creator
=========================

This module allows users to simply create neural networks by adding the
types of layers needed.

Example
-------

This is an example of how to use this module on a the classic XOR
Problem

.. code:: python

   import numpy as np

   from custom_neural_net_creator.model import Model
   from custom_neural_net_creator.dense import Dense
   from custom_neural_net_creator.activation_layer import ActivationLayer
   from custom_neural_net_creator.activation_functions import relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_prime
   from custom_neural_net_creator.loss_functions import mean_squared_error, mean_squared_error_derivative

   #Input data for XOR
   x = np.array([[0,0], [0,1], [1,0], [1,1]])
   y = np.array([[0], [1], [1], [0]])

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