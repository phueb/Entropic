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
from entropic.eval import calc_ba, make_p_cat
from entropic.corpus import Corpus
from entropic.rnn import RNN
from entropic.eval import softmax, get_windows


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
    sample_b = attr.ib(validator=attr.validators.instance_of(tuple))
    sample_a = attr.ib(validator=attr.validators.instance_of(tuple))
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

    save_path = Path(param2val['save_path'])

    # create toy input
    corpus = Corpus(doc_size=params.doc_size,
                    delay=params.delay,
                    num_types=params.num_types,
                    num_fragments=params.num_fragments,
                    period_probability=params.period_probability,
                    num_sentinels=params.num_sentinels,
                    sample_b=params.sample_b,
                    sample_a=params.sample_a,
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

    # check that ll x words are in prep.store
    assert len([p for p in corpus.x if p in prep.store.w2id]) == corpus.num_x

    # make helper dicts using IDs assigned to x-words by Preppy
    cat_id2xw_ids = {cat_id: [corpus.x.index(xw) for xw in corpus.cat_id2x[cat_id]]
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

            # get output representations
            x_v = np.array([[prep.store.w2id[ai]] for ai in corpus.a])
            x_w = np.array([[prep.store.w2id[bi]] for bi in corpus.b])
            x_x = np.array([[prep.store.w2id[xi]] for xi in corpus.x])
            x_y = np.array([[prep.store.w2id[yi]] for yi in corpus.y])
            q_v = softmax(rnn(torch.cuda.LongTensor(x_v))['logits'].detach().cpu().numpy())
            q_w = softmax(rnn(torch.cuda.LongTensor(x_w))['logits'].detach().cpu().numpy())
            q_x = softmax(rnn(torch.cuda.LongTensor(x_x))['logits'].detach().cpu().numpy())
            q_y = softmax(rnn(torch.cuda.LongTensor(x_y))['logits'].detach().cpu().numpy())

            # collect ba for all slots
            for slot, words in zip(corpus.slots,
                                   [corpus.a, corpus.x, corpus.b, corpus.y]):
                slot_id = corpus.slots.index(slot)
                print(f'slot={slot}')

                for num_left in range(slot_id + 1):
                    context_size = num_left + 1
                    print(f'\tcs={context_size}')

                    embeddings = []
                    for w in words:
                        # get windows for word
                        windows = get_windows(prep, [w], col_id=slot_id)
                        x = np.unique(windows, axis=0)[:, slot_id-num_left:slot_id+1]
                        # get embedding for word
                        inputs = torch.cuda.LongTensor(x)
                        embedding = rnn(inputs)['last_encodings'].detach().cpu().numpy().mean(axis=0)
                        embeddings.append(embedding)

                    sim_mat = cosine_similarity(np.vstack(embeddings))
                    ba = calc_ba(sim_mat, corpus.sim_mat_gold)
                    print(f'\t\tba={ba:.2f}')
                    name2col.setdefault(f'ba_{slot}_context-size={context_size}', []).append(ba)

            # collect dp between output-layer X sub-category representations
            eval_steps.append(step)
            if config.Eval.calc_dp:
                for cat_id1, cat_id2 in product(range(params.num_fragments), range(params.num_fragments)):
                    p_cat1 = make_p_cat(prep, x=corpus.cat_id2x[cat_id1], types=corpus.types)
                    q_cat2 = q_x[cat_id2xw_ids[cat_id2]].mean(0)
                    dp = drv.divergence_jensenshannon_pmf(p_cat1, q_cat2)
                    name2col.setdefault(f'dp_cat{cat_id1}_vs_cat{cat_id2}', []).append(dp)

            # collect pp
            pp = torch.exp(xe).detach().cpu().numpy().item()
            name2col.setdefault('pp', []).append(pp)

            # console
            print(f'step={step:>6,}/{prep.num_mbs:>6,}: pp={pp:.1f}', flush=True)
            print()

            # collect entropy of output-layer category representations
            for cat_id in range(params.num_fragments):
                q_cat = q_x[cat_id2xw_ids[cat_id]].mean(0)
                e = drv.entropy_pmf(q_cat)
                name2col.setdefault(f'e_cat{cat_id}', []).append(e)

            # save output probabilities to file (for making animations)
            if save_path.exists() and config.Eval.save_output_probabilities:  # does not exist when "ludwig -l"
                np.save(save_path / f'output_probabilities_v_{step:0>9}.npy', q_v)
                np.save(save_path / f'output_probabilities_w_{step:0>9}.npy', q_w)
                np.save(save_path / f'output_probabilities_x_{step:0>9}.npy', q_x)
                np.save(save_path / f'output_probabilities_y_{step:0>9}.npy', q_y)

            # save embeddings for x-word to file (for making animations)
            out_path = save_path / f'embeddings_{step:0>9}.npy'
            if save_path.exists() and config.Eval.save_embeddings:  # does not exist when "ludwig -l"
                for slot, words in zip(corpus.slots,
                                       [corpus.a, corpus.b, corpus.x, corpus.y]):
                    word_ids = [prep.store.w2id[w] for w in words]
                    embeddings = rnn.embed.weight.detach().cpu().numpy()[word_ids]
                    np.save(out_path, embeddings)

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
