import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, w1, w2):
        super(Net, self).__init__()

        # ops
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

        # custom weight init
        w1 = torch.from_numpy(w1.T.astype(np.float32))  # needs to be [output_size, input_size]
        w2 = torch.from_numpy(w2.T.astype(np.float32))  # needs to be [output_size, input_size]
        self.linear1.weight.data = nn.Parameter(w1)
        self.linear2.weight.data = nn.Parameter(w2)

    def forward(self, x):
        # layer 1 (hidden)
        layer1_z = self.linear1(x)
        layer1_a = torch.sigmoid(layer1_z)
        # layer 2 (output)
        layer2_z = self.linear2(layer1_a)
        layer2_a = torch.sigmoid(layer2_z)
        return layer2_a