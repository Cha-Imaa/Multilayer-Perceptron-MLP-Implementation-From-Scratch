import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z)) 
        return self.A

    def backward(self):
        dAdZ = self.A - self.A * self.A   
        return dAdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        self.A = np.tanh(Z) 
        return self.A

    def backward(self):
        dAdZ = (1 - self.A**2) 
        return dAdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """
    def forward(self, Z):
        self.A = np.maximum(0, Z) # TODO
        return self.A
    
    def backward(self):
        dAdZ = np.array(self.A > 0, dtype=int)
        return dAdZ
