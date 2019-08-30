import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, num_in, num_hidden, num_out, w1=None, w2=None):
        super(NeuralNet, self).__init__()
        # parameters
        self.num_in = num_in
        self.num_out = num_out
        self.num_hidden = num_hidden

        # weights
        # TODO experiment with weight init

        if w1 is None:
            self.w1 = torch.randn(self.num_in, self.num_hidden)
        else:
            self.w1 = torch.from_numpy(w1.astype(np.float32))

        if w2 is None:
            self.w2 = torch.randn(self.num_hidden, self.num_out)
        else:
            self.w2 = torch.from_numpy(w2.astype(np.float32))

    def forward(self, x):
        self.z = torch.matmul(x, self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = torch.matmul(self.z2, self.w2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return torch.scalar_tensor(1) / (1 + torch.exp(-s))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

    def backward(self, x, y, o):
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoid_derivative(o)  # derivative of sig to error
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.w2))
        self.z2_delta = self.z2_error * self.sigmoid_derivative(self.z2)
        self.w1 += torch.matmul(torch.t(x), self.z2_delta)
        self.w2 += torch.matmul(torch.t(self.z2), self.o_delta)

    def predict(self, x):
        return self.forward(torch.from_numpy(x)).numpy()