class BaseLayer:
    def __init__(self):
        self.x = None
        self.y

    def forward_prop(self, x):
        raise NotImplementedError
    
    def backward_prop(self, error, learning_rate):
        raise NotImplementedError