from engine import Tensor, Value
import numpy as np


class Module:
    """
    Base class for every layer.
    """

    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        pass

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        # Create Linear Module
        super(Linear, self).__init__()
        stdv = 1./np.sqrt(in_features)
        self.w = np.random.uniform(-stdv, stdv, size = (out_features, in_features)) 
        self.b = np.random.uniform(-stdv, stdv, size = out_features)
        
        self.gradw = np.zeros_like(self.w)
        self.gradb = np.zeros_like(self.b)
        
    def forward(self, inp):
        """Y = W * x + b"""
        self.output = np.dot(inp, self.w.T) + self.b
        return self.output 

    def parameters(self):
        # return 1-d list of all parameters List[Value]
        return [self.w, self.b]


class ReLU(Module):
    """The most simple and popular activation function"""
       def __init__(self):
            super(ReLu, self).__init__()
            
    def forward(self, inp):
        # Create ReLU Module
        self.output = np.maximum(inp,0)
        return self.output


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()
    
    def forward(self, inp, label):
        # Create CrossEntropy Loss Module
        self.output = - np.mean(inp[np.arange(label.shape[0]), label])
        return self.output
        
