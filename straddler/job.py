import attr
import numpy as np
import torch
import sys
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim as optim
import pandas as pd
from pathlib import Path

from straddler.data import Data
from straddler import config
from straddler.eval import calc_cluster_score
from straddler.net import Net
from straddler.utils import to_eval_epochs


@attr.s
class Params(object):
    batch_size = attr.ib(validator=attr.validators.instance_of(int))  # TODO params


    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'project_path', 'save_path']}
        return cls(**kwargs)


def main(param2val):

    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    # data
    data = Data(params)

    # net
    net = Net(params, data)

    # optimizer + criterion
    optimizer = optim.SGD(net.parameters(), lr=params.lr)
    criterion = torch.nn.MSELoss()  # TODO cross entropy

    # eval before start of training
    net.eval()
    torch_o = net(data.torch_x)
    torch_y = torch.from_numpy(data.make_y(epoch=0))
    loss = criterion(torch_o, torch_y)

    # train loop
    eval_epoch_idx = 0
    scores_a = np.zeros(config.Eval.num_evals)
    scores_b = np.zeros(config.Eval.num_evals)
    eval_epochs = to_eval_epochs(params)
    for epoch in range(params.num_epochs + 1):

        # eval
        if epoch in eval_epochs:
            # cluster score
            net.eval()
            print('Evaluating at epoch {}'.format(epoch))
            sim_mat = cosine_similarity(rep_mat)
            sim_mat_gold = np.rint(cosine_similarity(rep_mat_gold))
            score = calc_cluster_score(sim_mat, sim_mat_gold, config.Eval.metric)
            eval_epoch_idx += 1

            # mse
            mse = loss.detach().numpy().item()
            print('pp={}'.format(mse), flush=True)
            print()

        # get batch
        torch_x = torch.from_numpy(x)
        torch_y = torch.from_numpy(y)

        # train
        net.train()
        optimizer.zero_grad()  # zero the gradient buffers
        torch_o = net(torch_x)  # feed-forward
        loss = criterion(torch_o, torch_y)  # compute loss
        loss.backward()  # back-propagate
        optimizer.step()  # update

    # return performance as pandas Series # TODO return series not df
    s_a = pd.Series(scores_a, index=eval_epochs)
    s_b = pd.Series(scores_b, index=eval_epochs)
    s_a.name = 'results_a'
    s_b.name = 'results_b'

    return s_a, s_b