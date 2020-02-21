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
                       num_types=None,  # None ensures that no OOV symbol is inserted and all types are represented
                       slide_size=params.slide_size,
                       batch_size=params.batch_size,
                       context_size=1)

    xw_ids = [prep.store.w2id[xw] for xw in toy_corpus.xws]
    p_xw0 = make_xw_p(prep, prep.token_ids_array, toy_corpus.xws[1])

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
    e1s = []
    e2s = []
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

            # compute dp for Random x-word
            x_xw0 = np.array([[prep.store.w2id[toy_corpus.xws[0]]]])
            logit_xw0 = rnn(torch.cuda.LongTensor(x_xw0))['logits'].detach().cpu().numpy()[np.newaxis, :]
            q_xw0 = np.squeeze(softmax(logit_xw0))
            dp = drv.divergence_jensenshannon_pmf(p_xw0, q_xw0, base=np.exp(1).item())

            x_xw1 = np.array([[prep.store.w2id[toy_corpus.xws[1]]]])
            logit_xw1 = rnn(torch.cuda.LongTensor(x_xw1))['logits'].detach().cpu().numpy()[np.newaxis, :]
            q_xw1 = np.squeeze(softmax(logit_xw1))

            # TODO do next-word probabilities go up equally? if so, evidence of intermediate abstract category
            # get only probabilities for y-words
            q_xw0_yws = q_xw0[[prep.store.w2id[yw] for yw in toy_corpus.yws]]
            q_xw1_yws = q_xw1[[prep.store.w2id[yw] for yw in toy_corpus.yws]]
            print(q_xw0_yws[:8].round(6))
            print(q_xw1_yws[:8].round(6))

            # entropy of distribution over yws - should peak during early training - evidence of intermediate category
            e1 = drv.entropy_pmf(q_xw0_yws / sum(q_xw0_yws))
            e2 = drv.entropy_pmf(q_xw1_yws / sum(q_xw1_yws))
            print(e1)
            print(e2)

            # ba
            xw_reps = rnn.embed.weight.detach().cpu().numpy()[xw_ids]
            sim_mat = cosine_similarity(xw_reps)
            ba = calc_cluster_score(sim_mat, toy_corpus.sim_mat_gold, 'ba')

            # console
            pp = torch.exp(xe).detach().cpu().numpy().item()
            print(f'step={step:>6,}/{prep.num_mbs:>6,}: xe={xe:.1f} pp={pp:.1f} ba={ba:.4f} dp={dp:.4f}', flush=True)
            print()

            # collect performance data
            eval_steps.append(step)
            dps.append(dp)
            pps.append(pp)
            bas.append(ba)
            e1s.append(e1)
            e2s.append(e2)

        # TRAIN
        xe.backward()
        optimizer.step()

    # return performance as pandas Series
    s1 = pd.Series(dps, index=eval_steps)
    s2 = pd.Series(pps, index=eval_steps)
    s3 = pd.Series(bas, index=eval_steps)
    s4 = pd.Series(e1s, index=eval_steps)
    s5 = pd.Series(e2s, index=eval_steps)
    s1.name = 'dp'
    s2.name = 'pp'
    s3.name = 'ba'
    s4.name = 'e1'
    s5.name = 'e2'

    return s1, s2, s3, s4, s5