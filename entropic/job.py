import attr
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pyitlib import discrete_random_variable as drv

from preppy import SlidingPrep

from entropic import config
from entropic.eval import calc_cluster_score, make_xw_p
from entropic.toy_corpus import ToyCorpus
from entropic.rnn import RNN
from entropic.eval import softmax


@attr.s
class Params(object):
    # rnn
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    # toy corpus
    doc_size = attr.ib(validator=attr.validators.instance_of(int))
    num_xws = attr.ib(validator=attr.validators.instance_of(int))
    num_types = attr.ib(validator=attr.validators.instance_of(int))
    num_fragments = attr.ib(validator=attr.validators.instance_of(int))
    fragmentation_prob = attr.ib(validator=attr.validators.instance_of(float))
    # training
    slide_size = attr.ib(validator=attr.validators.instance_of(int))
    optimizer = attr.ib(validator=attr.validators.instance_of(str))
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))

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

    xw_ids = [prep.store.w2id[xw] for xw in toy_corpus.xws]
    p2 = make_xw_p(prep, prep.token_ids_array, toy_corpus.xws[1])

    rnn = RNN('srn', input_size=params.num_types, hidden_size=params.hidden_size)

    criterion = torch.nn.CrossEntropyLoss()
    if params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(rnn.parameters(), lr=params.lr)
    elif params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(rnn.parameters(), lr=params.lr)
    else:
        raise AttributeError('Invalid arg to "optimizer')

    # train loop
    eval_steps = []
    dps1 = []
    pps = []
    bas = []
    for step, batch in enumerate(prep.generate_batches()):

        # prepare x, y
        batch = batch[::params.num_fragments]  # get only windows where x is in first slot
        assert batch.shape[1] == 2
        x, y = batch[:, 0, np.newaxis], batch[:, 1]
        for xi in x[::2]:
            assert xi.item() in xw_ids
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(y)

        # feed forward
        rnn.train()
        optimizer.zero_grad()  # zero the gradient buffers
        logits = rnn(inputs)['logits']
        xe = criterion(logits, targets)

        # EVAL
        if step % config.Eval.eval_interval == 0:

            x2 = np.array([[prep.store.w2id[toy_corpus.xws[1]]]])
            logits2 = rnn(torch.cuda.LongTensor(x2))['logits'].detach().cpu().numpy()[np.newaxis, :]
            q2 = np.squeeze(softmax(logits2))

            dp1 = drv.divergence_jensenshannon_pmf(p2, q2, base=np.exp(1).item())

            print(f'{dp1:.4f}')

            # ba
            xw_reps = rnn.embed.weight.detach().cpu().numpy()[xw_ids]
            sim_mat = cosine_similarity(xw_reps)
            ba = calc_cluster_score(sim_mat, toy_corpus.sim_mat_gold, 'ba')

            # console
            pp = torch.exp(xe).detach().cpu().numpy().item()
            print(f'step={step:>6,}/{prep.num_mbs:>6,}: xe={xe:.1f} pp={pp:.1f} ba={ba:.4f} dp={dp1:.4f}', flush=True)
            print()

            # collect performance data
            eval_steps.append(step)
            dps1.append(dp1)
            pps.append(pp)
            bas.append(ba)

        # TRAIN
        xe.backward()
        optimizer.step()

    # return performance as pandas Series
    s1 = pd.Series(dps1, index=eval_steps)
    s2 = pd.Series(pps, index=eval_steps)
    s3 = pd.Series(bas, index=eval_steps)
    s1.name = 'dp'
    s2.name = 'pp'
    s3.name = 'ba'

    return s1, s2, s3