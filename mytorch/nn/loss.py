import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (self.A - self.Y) ** 2  # TODO
        sse = np.sum(se)  # TODO
        mse = sse / (2* self.N * self.C)  # TODO

        return mse

    def backward(self):

        dLdA = (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO

        # Ones_C = None  # TODO
        # Ones_N = None  # TODO

        exp_A = np.exp(self.A - np.max(self.A, axis=1, keepdims=True)) 
        self.softmax = exp_A / np.sum(exp_A, axis=1, keepdims=True)  # TODO
        crossentropy = self.Y * np.log(self.softmax + 1e-8)  # TODO
        sum_crossentropy = -np.sum(crossentropy) # TODO
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = self.softmax - self.Y  # TODO

        return dLdA
