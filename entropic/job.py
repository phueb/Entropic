import attr
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from pyitlib import discrete_random_variable as drv
from pathlib import Path
from itertools import product


from preppy import SlidingPrep

from entropic import config
from entropic.eval import calc_cluster_score, make_xw_true_out_probabilities
from entropic.corpus import Corpus
from entropic.rnn import RNN
from entropic.eval import softmax


@attr.s
class Params(object):
    # rnn
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    # toy corpus
    doc_size = attr.ib(validator=attr.validators.instance_of(int))
    delay = attr.ib(validator=attr.validators.instance_of(int))
    num_sentinels = attr.ib(validator=attr.validators.instance_of(int))
    num_types = attr.ib(validator=attr.validators.instance_of(int))
    num_fragments = attr.ib(validator=attr.validators.instance_of(int))
    period_probability = attr.ib(validator=attr.validators.instance_of(tuple))
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
    corpus = Corpus(doc_size=params.doc_size,
                    delay=params.delay,
                    num_types=params.num_types,
                    num_fragments=params.num_fragments,
                    period_probability=params.period_probability,
                    num_sentinels=params.num_sentinels,
                    )
    prep = SlidingPrep([corpus.doc],
                       reverse=False,
                       num_types=None,  # None ensures that no OOV symbol is inserted and all types are represented
                       slide_size=params.slide_size,
                       batch_size=params.batch_size,
                       context_size=corpus.num_words_in_window-1)

    # check that types in corpus and prep are identically ordered
    for t1, t2, in zip(prep.store.types, corpus.types):
        assert t1 == t2

    # make helper dicts using IDs assigned to x-words by Preppy
    xw_ids = [prep.store.w2id[xw] for xw in corpus.x]
    cat_id2xw_ids = {cat_id: [corpus.x.index(xw) for xw in corpus.cat_id2x[cat_id]]
                     for cat_id in range(params.num_fragments)}
    cat_id2p = {cat_id: make_xw_true_out_probabilities(prep, x=corpus.cat_id2x[cat_id], types=corpus.types)
                for cat_id in range(params.num_fragments)}

    rnn = RNN('srn', input_size=params.num_types, hidden_size=params.hidden_size)

    criterion = torch.nn.CrossEntropyLoss()
    if params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(rnn.parameters(), lr=params.lr)
    elif params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(rnn.parameters(), lr=params.lr)
    else:
        raise AttributeError('Invalid arg to "optimizer')

    eval_steps = []
    name2col = {}

    # train loop
    for step, batch in enumerate(prep.generate_batches()):

        # remove y-words from slot 1
        if params.xws_in_slot_1_only:
            batch = batch[::params.num_fragments]  # get only windows where x is in first slot
            assert batch[0, 0].item() in xw_ids

        # prepare x, y
        x, y = batch[:, :-1], batch[:, -1]
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
            x_xws = np.array([[prep.store.w2id[xw]] for xw in corpus.x])
            q_x = softmax(rnn(torch.cuda.LongTensor(x_xws))['logits'].detach().cpu().numpy())

            # ba
            embeddings_xws = rnn.embed.weight.detach().cpu().numpy()[xw_ids]
            if config.Eval.calc_ba:
                sim_mat = cosine_similarity(embeddings_xws)
                ba = calc_cluster_score(sim_mat, corpus.sim_mat_gold, 'ba')
            else:
                ba = np.nan

            # console
            pp = torch.exp(xe).detach().cpu().numpy().item()
            print(f'step={step:>6,}/{prep.num_mbs:>6,}: xe={xe:.1f} pp={pp:.1f} ba={ba:.4f}', flush=True)
            print()

            # collect dp between output-layer cat representations
            eval_steps.append(step)
            if config.Eval.calc_dp:
                for cat_id1, cat_id2 in product(range(params.num_fragments), range(params.num_fragments)):
                    p_cat1 = cat_id2p[cat_id1]
                    q_cat2 = q_x[cat_id2xw_ids[cat_id2]].mean(0)
                    dp = drv.divergence_jensenshannon_pmf(p_cat1, q_cat2)
                    name2col.setdefault(f'dp_cat{cat_id1}_vs_cat{cat_id2}', []).append(dp)

            # collect pp + ba
            name2col.setdefault('pp', []).append(pp)
            name2col.setdefault('ba', []).append(ba)

            # collect entropy of output-layer category representations
            for cat_id in range(params.num_fragments):
                q_cat = q_x[cat_id2xw_ids[cat_id]].mean(0)
                e = drv.entropy_pmf(q_cat)
                name2col.setdefault(f'e_cat{cat_id}', []).append(e)

            assert embeddings_xws.shape[0] == q_x.shape[0]

            # save output probabilities for x-word to file for making SVD time-course animation
            out_path = save_path / f'output_probabilities_{step:0>9}.npy'
            if save_path.exists():  # does not exist when running ludwig with --local
                np.save(out_path, q_x)

            # save embeddings for x-word to file for making SVD time-course animation
            out_path = save_path / f'embeddings_{step:0>9}.npy'
            if save_path.exists():  # does not exist when running ludwig with --local
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