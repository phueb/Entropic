

from pyitlib import discrete_random_variable as drv
from numpy.lib.stride_tricks import as_strided
import numpy as np
import matplotlib.pyplot as plt

from preppy import SlidingPrep
from ludwig.results import gen_all_param2vals

from entropic.corpus import Corpus
from entropic.job import Params
from entropic.params import param2default


NUM_TICKS = 32
DISTANCE = -1  # can be negative or positive
NORM_FACTOR = 'XY'


param2requests = {
                  'redundant_a': [([1.0, 0.0], [1.0, 1.0])],
                  }


def collect_data(windows_mat, reverse: bool):

    if reverse:
        windows_mat = np.flip(windows_mat, 0)

    res = []
    for num_windows in x_ticks:

        ws = windows_mat[:num_windows]

        # x-word windows
        x_loc = 1
        row_ids = np.isin(ws[:, x_loc], [prep.store.w2id[w] for w in probes])
        probe_windows = ws[row_ids]

        # mutual info
        # assert DISTANCE <= x_loc - 1
        x = probe_windows[:, x_loc + DISTANCE]  # left or right context
        y = probe_windows[:, x_loc]  # probe
        mi = drv.information_mutual_normalised(x, y, norm_factor=NORM_FACTOR)

        res.append(mi)

    return res


for param2val in gen_all_param2vals(param2requests, param2default):
    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)
    print()

    corpus = Corpus(doc_size=params.doc_size,
                    num_types=params.num_types,
                    num_fragments=params.num_fragments,
                    starvation=params.starvation,
                    redundant_a=params.redundant_a,
                    redundant_b=params.redundant_b,
                    size_a=params.size_a,
                    size_b=params.size_b,
                    drop_a=params.drop_a,
                    drop_b=params.drop_b,
                    )
    prep = SlidingPrep([corpus.sequences],
                       reverse=False,
                       num_types=None,  # None ensures that no OOV symbol is inserted and all types are represented
                       slide_size=params.batch_size,
                       batch_size=params.batch_size,
                       context_size=corpus.num_words_in_window - 1)

    probes = corpus.x
    print(f'num probes={len(probes)}')

    # windows
    token_ids_array = np.array(prep.store.token_ids, dtype=np.int64)
    num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
    print(f'Matrix containing all windows has shape={windows.shape}')

    x_ticks = [int(i) for i in np.linspace(0, len(windows), NUM_TICKS + 1)][1:]

    # collect data
    mi1 = collect_data(windows, reverse=False)
    mi2 = collect_data(windows, reverse=True)

    # fig
    fig, ax = plt.subplots(1, figsize=(6, 5), dpi=163)
    fontsize = 12
    plt.title(f'distance={DISTANCE}\nnorm={NORM_FACTOR}',
              fontsize=fontsize)
    ax.set_ylabel('Cumulative Normalized Mutual Info', fontsize=fontsize)
    ax.set_xlabel('Location in Corpus [num tokens]', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([i if n in [0, len(x_ticks) - 1] else '' for n, i in enumerate(x_ticks)])
    # plot
    ax.plot(x_ticks, mi1, '-', linewidth=2, color='C0', label='reverse=False')
    ax.plot(x_ticks, mi2, '-', linewidth=2, color='C1', label='reverse=True')
    plt.legend(frameon=False)
    plt.show()



