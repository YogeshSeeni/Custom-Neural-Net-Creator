class ActivationLayer():
    def __init__(self, activation_function, activation_derivative):
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward_prop(self, x):
        self.x = x
        #pass input into activation function
        self.y = self.activation_function(self.x)
        return self.y
    
    def backward_prop(self, dE_dY, learning_rate):
        #Return multiplication of activation derivative and ∂E/∂Y as there is not parameters to optimize in this layer
        return self.activation_derivative(self.x) * dE_dY
        