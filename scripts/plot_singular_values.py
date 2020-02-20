from numpy.lib.stride_tricks import as_strided
from pyitlib import discrete_random_variable as drv
import numpy as np
import matplotlib.pyplot as plt

from preppy import PartitionedPrep

from straddler.figs import make_example_fig
from straddler.toy_corpus import ToyCorpus
from straddler.figs import plot_singular_values

DOC_SIZE = 100_000
NUM_TYPES = 1024  # this needs to be large to provide room for interesting effects
NUM_XWS = 512
NUM_FRAGMENTS = 3  # number of x-word sub categories, or singular dimensions
FRAGMENTATION_PROB = 1.0  # the higher, the more prominent are sub-categories (and singular values)
NUM_SINGULAR_DIMS_PLOT = 8


toy_corpus = ToyCorpus(doc_size=DOC_SIZE,
                       num_types=NUM_TYPES,
                       num_xws=NUM_XWS,
                       num_fragments=NUM_FRAGMENTS,
                       fragmentation_prob=FRAGMENTATION_PROB,
                       )
prep = PartitionedPrep([toy_corpus.doc],
                       reverse=False,
                       num_types=NUM_TYPES,
                       num_parts=2,
                       num_iterations=[1, 1],
                       batch_size=64,
                       context_size=1)
probes = [p for p in toy_corpus.xws if p in prep.store.w2id]

s_list = []
for part_id, part in enumerate(prep.reordered_parts):

    # windows
    token_ids_array = part.astype(np.int64)
    num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

    # windows with probe in position -2
    row_ids = np.isin(windows[:, -2], [prep.store.w2id[w] for w in probes])
    probe_windows = windows[row_ids]

    # conditional entropy
    x = probe_windows[:, -2]  # x-word
    y = probe_windows[:, -1]  # y-word
    x_y = np.vstack((x, y))   # joint outcome

    # map word ID of x-words to IDs between [0, len(probes)]
    # this makes creating a matrix with the right number of columns easier
    x2x = {xi: n for n, xi in enumerate(np.unique(x))}

    # make co-occurrence plot
    cf_mat = np.ones((prep.num_types, len(x2x))) * 1e-9
    for xi, yi in zip(x, y):
        cf_mat[yi, x2x[xi]] += 1
    last_num_rows = NUM_TYPES - NUM_XWS  # other rows are just empty because of x-words not occurring with x-words
    fig, ax = make_example_fig(np.log(cf_mat[-last_num_rows:]))
    ce = drv.entropy_conditional(x, y).item()
    je = drv.entropy_joint(x_y).item()
    ye = drv.entropy_joint(y).item()
    plt.title(f'Toy Corpus Part {part_id+1}\nH(x-word|y-word)={ce:.4f}\nH(x-word,y-word)={je:.4f}\nH(y-word)={ye:.4f}')
    plt.show()

    # collect singular values for plotting
    s = np.linalg.svd(cf_mat, compute_uv=False)
    s_list.append(np.asarray(s[:NUM_SINGULAR_DIMS_PLOT]))

plot_singular_values(s_list, max_s=NUM_SINGULAR_DIMS_PLOT)

