import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):

    def __init__(self, input_size, output_size, w1):
        super(Net, self).__init__()
        # hyper-parameters
        self.num_in = input_size
        self.num_out = output_size

        # ops
        self.linear1 = nn.Linear(input_size, output_size)

        # custom weight init
        w1 = torch.from_numpy(w1.T.astype(np.float32))  # needs to be [output_size, input_size]
        self.linear1.weight.data = nn.Parameter(w1)

        # TODO debug
        # print(w1)
        # for p in self.parameters():
        #     print(p)

    def forward(self, x):
        h_z = self.linear1(x)
        h_a = torch.sigmoid(h_z)
        return h_a