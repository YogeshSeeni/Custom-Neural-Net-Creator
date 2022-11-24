class Model():
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        #Stores list of all outputs for each input
        output = []

        for i in range(len(input)):
            #Keeps track of current output for the ith input (By Default the first is the input layer)
            singleOutput = input[i]

            #Forward propogate each layer using the output of the last layer
            for layer in self.layers:
                singleOutput = layer.forward_prop(singleOutput)
            
            #Add to the output list
            output.append(singleOutput)

        return output

    def print_error(self, epoch, error):
        print(f"Loss of Epoch #{epoch}: {error}")

    def fit(self, x, y, loss, loss_derivative, epochs=1000, learning_rate=0.01, verbosity=1):
        #Needed for Backpropogation
        reversed_layers = reversed(self.layers)

        for i in range(epochs):
            #Keeps track of error for current epoch for potential display at the end (depends on level of verbosity)
            total_error = 0

            for j in range(len(x)):
                #Keeps track of current output (starts with the input layer)
                output = input[j]

                #Forward propogate each layer using the output of the last layer
                for layer in self.layers:
                    output = layer.forward_prop(output)

                #Update the total error
                total_error += loss(y[j], output)

                #Backpropogation: Update weights and biases for each layer using derivative of error with respect to the output
                dE_dY = loss_derivative(y[j], output)

                #Loop backwards over layers as ∂E/∂X of the last layer is ∂E/∂Y of the layer before it
                for layer in reversed_layers:
                    dE_dY = layer.backward_prop(dE_dY, learning_rate)

            average_error = total_error / len(x)

            if verbosity == 0:
                continue
            elif verbosity == 1:
                if (i+1) % 100 == 0:
                    self.print_error(i+1, average_error)
            elif verbosity == 2:
                if (i+1) % 10 == 0:
                    self.print_error(i+1, average_error)
            else:
                self.print_error(i+1, average_error)