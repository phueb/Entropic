import attr
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pyitlib import discrete_random_variable as drv

from preppy import SlidingPrep

from straddler import config
from straddler.eval import calc_cluster_score, make_straddler_p
from straddler.toy_corpus import ToyCorpus
from straddler.rnn import RNN
from straddler.eval import softmax


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
    straddler_p = make_straddler_p(prep, prep.token_ids_array, toy_corpus.straddler)

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
    dps = []
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
        loss = criterion(logits, targets)

        # EVAL
        if step % config.Eval.eval_interval == 0:

            # dp - how lexically specific are next word predictions for straddler, which is in no particular sub-group?
            straddler_x = np.array([[prep.store.w2id[toy_corpus.straddler]]])
            straddler_logits = rnn(torch.cuda.LongTensor(straddler_x))['logits'].detach().cpu().numpy()[np.newaxis, :]
            straddler_q = np.squeeze(softmax(straddler_logits))

            print(straddler_p.shape)
            print(straddler_q.shape)

            dp = drv.divergence_jensenshannon_pmf(straddler_p, straddler_q)

            # ba
            xw_reps = rnn.embed.weight.detach().cpu().numpy()[xw_ids]
            sim_mat = cosine_similarity(xw_reps)
            ba = calc_cluster_score(sim_mat, toy_corpus.sim_mat_gold, 'ba')

            # console
            pp = torch.exp(loss).detach().cpu().numpy().item()
            print(f'step={step:>6,}/{prep.num_mbs}: pp={pp:.1f} ba={ba:.4f} dp={dp:.4f}', flush=True)
            print()

            # collect performance data
            eval_steps.append(step)
            dps.append(dp)
            pps.append(pp)
            bas.append(ba)

        # TRAIN
        loss.backward()
        optimizer.step()

    # return performance as pandas Series
    s1 = pd.Series(dps, index=eval_steps)
    s2 = pd.Series(pps, index=eval_steps)
    s3 = pd.Series(bas, index=eval_steps)
    s1.name = 'dp_straddler'
    s2.name = 'pp'
    s3.name = 'ba'

    return s1, s2, s3