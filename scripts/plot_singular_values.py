from pyitlib import discrete_random_variable as drv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, scale

from preppy import PartitionedPrep

from entropic.figs import make_heatmap_fig
from entropic.toy_corpus import ToyCorpus
from entropic.figs import plot_singular_values
from entropic.outcomes import get_outcomes

DOC_SIZE = 500_000
NUM_TYPES = 1024  # this needs to be large to provide room for interesting effects
NUM_XWS = 512
NUM_FRAGMENTS = 4  # number of x-word sub categories, or singular dimensions
ALPHA = 2.0
PERIOD_PROBABILITIES = [0.0]  # TODO
NUM_SINGULAR_DIMS_PLOT = 8

SHOW_HEATMAP = True


s_list_scaled = []
s_list_intact = []
for pp in PERIOD_PROBABILITIES:
    toy_corpus = ToyCorpus(doc_size=DOC_SIZE,
                           num_types=NUM_TYPES,
                           num_xws=NUM_XWS,
                           num_fragments=NUM_FRAGMENTS,
                           alpha=ALPHA,
                           period_probability=pp
                           )
    prep = PartitionedPrep([toy_corpus.doc],
                           reverse=False,
                           num_types=None,
                           num_parts=2,
                           num_iterations=[1, 1],
                           batch_size=64,
                           context_size=1)
    probes = [p for p in toy_corpus.xws if p in prep.store.w2id]

    # get outcomes - the word IDs that occur in the same 2-word window
    x, y, x_y = get_outcomes(prep, prep.store.token_ids, probes)

    # map word ID of x-words to IDs between [0, len(probes)]
    # this makes creating a matrix with the right number of columns easier
    x2x = {xi: n for n, xi in enumerate(np.unique(x))}

    # make co-occurrence matrix
    cf_mat = np.ones((prep.num_types, len(x2x))) * 1e-9
    for xi, yi in zip(x, y):
        cf_mat[yi, x2x[xi]] += 1

    # make co-occurrence plot
    if SHOW_HEATMAP:
        last_num_rows = NUM_TYPES - NUM_XWS  # other rows are just empty because of x-words not occurring with x-words
        fig, ax = make_heatmap_fig(np.log(cf_mat[-last_num_rows:]))
        ce = drv.entropy_conditional(x, y).item()
        je = drv.entropy_joint(x_y).item()
        ye = drv.entropy_joint(y).item()
        plt.title(f'Toy Corpus\nperiod prob={pp}\nH(x-word|y-word)={ce:.4f}\nH(x-word,y-word)={je:.4f}\nH(y-word)={ye:.4f}')
        plt.show()

    # collect singular values for plotting
    cf_mat_intact = scale(cf_mat, axis=1, with_std=False, with_mean=False)
    cf_mat_scaled = scale(cf_mat, axis=1, with_std=False, with_mean=True)  # subtracting mean from rows
    s_intact = np.linalg.svd(cf_mat_intact, compute_uv=False)
    s_scaled = np.linalg.svd(cf_mat_scaled, compute_uv=False)
    s_list_intact.append(np.asarray(s_intact[:NUM_SINGULAR_DIMS_PLOT]))
    s_list_scaled.append(np.asarray(s_scaled[:NUM_SINGULAR_DIMS_PLOT]))

# difference between normalizing and no normalizing matters!
plot_singular_values(s_list_intact, scaled=False, max_s=NUM_SINGULAR_DIMS_PLOT, pps=PERIOD_PROBABILITIES)
plot_singular_values(s_list_scaled, scaled=True, max_s=NUM_SINGULAR_DIMS_PLOT, pps=PERIOD_PROBABILITIES)

