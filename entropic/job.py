import attr
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pyitlib import discrete_random_variable as drv
from pathlib import Path

from preppy import SlidingPrep

from entropic import config
from entropic.eval import calc_cluster_score, make_xw_true_out_probabilities
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
    period_probability = attr.ib(validator=attr.validators.instance_of(float))
    # training
    xws_in_slot_1_only = attr.ib(validator=attr.validators.instance_of(bool))
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

    save_path = Path(param2val['save_path'])

    # create toy input
    toy_corpus = ToyCorpus(doc_size=params.doc_size,
                           num_types=params.num_types,
                           num_xws=params.num_xws,
                           num_fragments=params.num_fragments,
                           period_probability=params.period_probability,
                           )
    prep = SlidingPrep([toy_corpus.doc],
                       reverse=False,
                       num_types=None,  # None ensures that no OOV symbol is inserted and all types are represented
                       slide_size=params.slide_size,
                       batch_size=params.batch_size,
                       context_size=1)

    xw_ids = [prep.store.w2id[xw] for xw in toy_corpus.xws]
    p_xw0 = make_xw_true_out_probabilities(prep, prep.token_ids_array, toy_corpus.xws[0])  # this xw is in category 1
    p_xw1 = make_xw_true_out_probabilities(prep, prep.token_ids_array, toy_corpus.xws[1])  # this xw is in category 2

    rnn = RNN('srn', input_size=params.num_types, hidden_size=params.hidden_size)

    criterion = torch.nn.CrossEntropyLoss()
    if params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(rnn.parameters(), lr=params.lr)
    elif params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(rnn.parameters(), lr=params.lr)
    else:
        raise AttributeError('Invalid arg to "optimizer')

    eval_steps = []
    name2col = {
        'dp_0_0': [],
        'dp_0_1': [],
        'dp_1_1': [],
        'dp_1_0': [],
        'e1': [],
        'e2': [],
        'sing-dim-1_0': [],  # xw 0 loading on sing dim 1
        'sing-dim-2_0': [],  # xw 0 loading on sing dim 2
        'sing-dim-1_1': [],  # xw 1 loading on sing dim 1
        'sing-dim-2_1': [],  # xw 1 loading on sing dim 1
        'ba': [],
        'pp': [],
    }
    # train loop
    for step, batch in enumerate(prep.generate_batches()):

        # TODO this determines whether phantom category emerges or not - why?
        if params.xws_in_slot_1_only:
            batch = batch[::params.num_fragments]  # get only windows where x is in first slot
            assert batch[0, 0].item() in xw_ids

        # prepare x, y
        x, y = batch[:, 0, np.newaxis], batch[:, 1]
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(y)

        # feed forward
        rnn.train()
        optimizer.zero_grad()  # zero the gradient buffers
        logits = rnn(inputs)['logits']
        xe = criterion(logits, targets)

        # EVAL
        if step % config.Eval.eval_interval == 0:

            # get output representations for all x-words
            x_xws = np.array([[prep.store.w2id[xw]] for xw in toy_corpus.xws])
            output_probabilities_xws = softmax(rnn(torch.cuda.LongTensor(x_xws))['logits'].detach().cpu().numpy())

            # compute dp between xw 1 and 0 and vice versa
            q_xw0 = output_probabilities_xws[0]
            q_xw1 = output_probabilities_xws[1]
            dp_0_0 = drv.divergence_jensenshannon_pmf(p_xw0, q_xw0)
            dp_0_1 = drv.divergence_jensenshannon_pmf(p_xw0, q_xw1)
            dp_1_1 = drv.divergence_jensenshannon_pmf(p_xw1, q_xw1)
            dp_1_0 = drv.divergence_jensenshannon_pmf(p_xw1, q_xw0)

            # show that cat 1 and 2 representations converge along sing dim 1 together at first,
            #  and then diverge along sing dim 2
            u, s, v = np.linalg.svd(output_probabilities_xws, compute_uv=True)

            # singular dim 2 should increase steadily when num_fragments=2
            print(f'dim1 sv={s[0]:2.4f}')
            print(f'dim2 sv={s[1]:2.4f}')
            print(f'dim3 sv={s[2]:2.4f}')
            print(f'dim4 sv={s[3]:2.4f}')

            # get only probabilities for y-words
            q_xw0_yws = q_xw0[[prep.store.w2id[yw] for yw in toy_corpus.yws]]
            q_xw1_yws = q_xw1[[prep.store.w2id[yw] for yw in toy_corpus.yws]]
            # entropy of distribution over yws - should peak during early training - evidence of intermediate category
            e1 = drv.entropy_pmf(q_xw0_yws / sum(q_xw0_yws))
            e2 = drv.entropy_pmf(q_xw1_yws / sum(q_xw1_yws))

            # entropy should move from 9 to 8 if using num_fragments=2 and num_types=1024, because 2^8=1024/2
            print(e1)
            print(e2)

            # ba
            embeddings_xws = rnn.embed.weight.detach().cpu().numpy()[xw_ids]
            sim_mat = cosine_similarity(embeddings_xws)
            ba = calc_cluster_score(sim_mat, toy_corpus.sim_mat_gold, 'ba')

            # console
            pp = torch.exp(xe).detach().cpu().numpy().item()
            print(f'step={step:>6,}/{prep.num_mbs:>6,}: xe={xe:.1f} pp={pp:.1f} ba={ba:.4f}', flush=True)
            print()

            # collect performance data
            eval_steps.append(step)
            name2col['dp_0_0'].append(dp_0_0)
            name2col['dp_0_1'].append(dp_0_1)
            name2col['dp_1_1'].append(dp_1_1)
            name2col['dp_1_0'].append(dp_1_0)
            name2col['pp'].append(pp)
            name2col['ba'].append(ba)
            name2col['e1'].append(e1)
            name2col['e2'].append(e2)
            name2col['sing-dim-1_0'].append(u[0::2, 0].mean())  # this is informative only when num_fragments=2
            name2col['sing-dim-2_0'].append(u[0::2, 1].mean())
            name2col['sing-dim-1_1'].append(u[1::2, 0].mean())
            name2col['sing-dim-2_1'].append(u[1::2, 1].mean())

            assert embeddings_xws.shape[0] == output_probabilities_xws.shape[0]

            # save output probabilities for x-word to file for making SVD time-course animation
            out_path = save_path / f'output_probabilities_{step:0>9}.npy'
            np.save(out_path, output_probabilities_xws)

            # save embeddings for x-word to file for making SVD time-course animation
            out_path = save_path / f'embeddings_{step:0>9}.npy'
            np.save(out_path, embeddings_xws)

        # TRAIN
        xe.backward()
        optimizer.step()

    # return performance as pandas Series
    series_list = []
    for name, col in name2col.items():
        print(f'Making pandas series with name={name} and length={len(col)}')
        s = pd.Series(col, index=eval_steps)
        s.name = name
        series_list.append(s)

    return series_list