import attr
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim as optim
import pandas as pd

from preppy import SlidingPrep

from straddler import config
from straddler.eval import calc_cluster_score
from straddler.toy_corpus import ToyCorpus
from straddler.rnn import RNN


@attr.s
class Params(object):
    # rnn
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    # toy corpus
    doc_size = attr.ib(validator=attr.validators.instance_of(int))
    num_xws = attr.ib(validator=attr.validators.instance_of(int))
    num_types = attr.ib(validator=attr.validators.instance_of(int))
    num_fragments = attr.ib(validator=attr.validators.instance_of(int))
    fragmentation_prob = attr.ib(validator=attr.validators.instance_of(float))
    # training
    slide_size = attr.ib(validator=attr.validators.instance_of(int))

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

    # create toy input
    toy_corpus = ToyCorpus(doc_size=params.doc_size,
                           num_types=params.num_types,
                           num_xws=params.num_xws,
                           num_fragments=params.num_fragments,
                           fragmentation_prob=params.fragmentation_prob,
                           )
    prep = SlidingPrep([toy_corpus.doc],
                       reverse=False,
                       num_types=params.num_types,
                       slide_size=params.slide_size,
                       batch_size=params.batch_size,
                       context_size=1)

    rnn = RNN('rnn', input_size=params.num_types, hidden_size=params.hidden_size)

    # optimizer + criterion
    optimizer = optim.SGD(rnn.parameters(), lr=params.lr)
    criterion = torch.nn.MSELoss()  # TODO cross entropy

    # train loop
    eval_steps = []
    dps = []
    pps = []
    bas = []
    for step, batch in enumerate(prep.generate_batches()):

        # prepare x, y
        x, y = batch[:-1], batch[:, -1]
        torch_x = torch.from_numpy(x)
        torch_y = torch.from_numpy(y)

        # ba
        rnn.eval()
        sim_mat = cosine_similarity(rep_mat)
        sim_mat_gold = np.rint(cosine_similarity(rep_mat_gold))
        ba = calc_cluster_score(sim_mat, sim_mat_gold, 'ba')

        # feed-forward + compute loss
        rnn.train()
        optimizer.zero_grad()  # zero the gradient buffers
        torch_o = rnn(torch_x)  # feed-forward
        loss = criterion(torch_o, torch_y).detach().numpy().item()  # compute loss

        # console
        pp = torch.exp(loss)
        print(f'pp={pp}', flush=True)
        print()

        # update RNN weights
        loss.backward()  # back-propagate
        optimizer.step()  # update

        # collect performance data
        eval_steps.append(step)
        dps.append(dp)
        pps.append(pp)
        bas.append(ba)

    # return performance as pandas Series
    s1 = pd.Series(dps, index=eval_steps)
    s2 = pd.Series(pps, index=eval_steps)
    s3 = pd.Series(bas, index=eval_steps)
    s1.name = 'dp_straddler'
    s2.name = 'pp'
    s3.name = 'ba'

    return s1, s2, s3