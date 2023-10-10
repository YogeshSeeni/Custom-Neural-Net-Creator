Quickstart
==========

Welcome to the quickstart guide for the `custom-neural-net-creator` module, a Python library for creating custom neural networks. In this guide, you will learn how to create a simple neural network for the XOR problem.

Prerequisites
-------------

Before you get started, ensure you have the following prerequisites installed:

- Python 3.x
- ``numpy`` library
- The ``custom-neural-net-creator`` module (make sure it's properly installed).

Importing Required Libraries and Modules
-----------------------------------------

To begin, import the necessary libraries and modules:

.. code-block:: python

   import numpy as np
   from custom_neural_net_creator.model import Model
   from custom_neural_net_creator.dense import Dense
   from custom_neural_net_creator.activation_layer import ActivationLayer
   from custom_neural_net_creator.activation_functions import (
       relu, relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_prime
   )
   from custom_neural_net_creator.loss_functions import (
       mean_squared_error, mean_squared_error_derivative
   )

Input Data for XOR Problem
--------------------------

Define the input data and target values for the XOR problem:

.. code-block:: python

   x = [[0,0], [0,1], [1,0], [1,1]]
   y = [[0], [1], [1], [0]]

Creating the Neural Network Model
-----------------------------------

Now, let's create the neural network model:

.. code-block:: python

   model = Model()

   # Adding the first hidden layer with 10 neurons and ReLU activation
   model.add(Dense(2, 10))
   model.add(ActivationLayer(relu, relu_derivative))

   # Adding the second hidden layer with 10 neurons and ReLU activation
   model.add(Dense(10, 10))
   model.add(ActivationLayer(relu, relu_derivative))

   # Adding the output layer with 1 neuron and Sigmoid activation
   model.add(Dense(10, 1))
   model.add(ActivationLayer(sigmoid, sigmoid_derivative))

Training the Model
-------------------

To train the model on the XOR problem, use the following code:

.. code-block:: python

   model.fit(x, y, mean_squared_error, mean_squared_error_derivative)

After training, you will see the loss for each epoch, and the final loss value.

Testing the Model
-----------------

To test the trained model, make predictions on a subset of the input data:

.. code-block:: python

   predictions = model.predict(x[0:3])

   print("Predicted: ")
   print(predictions)

You will get the model's predictions for the input data.

.. code-block:: plaintext

   Predicted:
   [array([[0.02610931]]), array([[0.98778214]]), array([[0.9873547]])]

Conclusion
----------

Congratulations! You have successfully created and trained a custom neural network using the `custom-neural-net-creator` module. You can now integrate this module into your own projects and experiments for more complex neural network tasks. Explore the module's documentation for advanced features and customization options.
