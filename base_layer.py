class BaseLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward_prop(self, x):
        raise NotImplementedError
    
    def backward_prop(self, dE_dY, learning_rate):
        raise NotImplementedError