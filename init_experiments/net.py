import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):

    def __init__(self, params, data):
        super(Net, self).__init__()
        self.params = params
        self.data = data

        # ops
        self.linear1 = nn.Linear(data.input_size, params.hidden_size)
        self.linear2 = nn.Linear(params.hidden_size, data.output_size)

        # custom weight init
        w1 = self.make_w1()
        w2 = np.random.standard_normal(size=(params.hidden_size, data.output_size))
        #
        torch_w1 = torch.from_numpy(w1.T.astype(np.float32))  # needs to be [output_size, input_size]
        torch_w2 = torch.from_numpy(w2.T.astype(np.float32))  # needs to be [output_size, input_size]
        self.linear1.weight.data = nn.Parameter(torch_w1)
        self.linear2.weight.data = nn.Parameter(torch_w2)

    def forward(self, x):
        # layer 1 (hidden)
        layer1_z = self.linear1(x)
        layer1_a = torch.sigmoid(layer1_z)
        # layer 2 (output)
        layer2_z = self.linear2(layer1_a)
        layer2_a = torch.sigmoid(layer2_z)
        return layer2_a

    def make_w1(self):
        if self.params.init == 'random':
            w1 = np.random.standard_normal(size=(self.data.input_size, self.params.hidden_size))
        #
        elif self.params.init == 'superordinate':
            a = np.tile(np.random.standard_normal(size=(1, self.params.hidden_size)), (self.data.input_size_a, 1))
            b = np.tile(np.random.standard_normal(size=(1, self.params.hidden_size)), (self.data.input_size_b, 1))
            w1 = np.vstack((a, b))  # [num_items_in_a + num_items_in_b, num_cols]
        #
        elif self.params.init == 'identical':
            w1 = np.tile(np.random.standard_normal(size=(1, self.params.hidden_size)), (self.data.input_size, 1))
        #
        elif self.params.init == 'linear':  # each item is assigned same weight vector with linear transformation
            w1 = np.tile(np.random.standard_normal(size=(1, self.params.hidden_size)), (self.data.input_size, 1))
            w1_delta = np.tile(np.linspace(0, 1, num=w1.shape[0])[:, np.newaxis], (1, w1.shape[1]))
            w1 += np.random.permutation(
                w1_delta)  # permute otherwise linear transform will align with category structure
        #
        else:
            raise AttributeError('Invalid arg to "init".')
        #
        return w1 * self.params.scale_weights