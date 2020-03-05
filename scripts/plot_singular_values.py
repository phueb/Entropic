from pyitlib import discrete_random_variable as drv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, scale

from preppy import PartitionedPrep

from entropic.figs import make_heatmap_fig
from entropic.corpus import Corpus
from entropic.figs import plot_singular_values
from entropic.outcomes import get_outcomes

DOC_SIZE = 400_000
DELAY = 200_000
NUM_TYPES = 512  # this needs to be large to provide room for interesting effects
NUM_FRAGMENTS = 4  # number of x-word sub categories, or singular dimensions
ALPHA = 2.0
PERIOD_PROBABILITY = (0.0, 0.0)
NUM_SENTINELS_LIST = [32, 16]
NUM_S_DIMS = 8

SHOW_HEATMAP = True


s_list_scaled = []
s_list_intact = []
for ns in NUM_SENTINELS_LIST:
    corpus = Corpus(doc_size=DOC_SIZE,
                    delay=DELAY,
                    num_types=NUM_TYPES,
                    num_fragments=NUM_FRAGMENTS,
                    alpha=ALPHA,
                    period_probability=PERIOD_PROBABILITY,
                    num_sentinels=ns,
                    )
    prep = PartitionedPrep([corpus.doc],
                           reverse=False,
                           num_types=None,
                           num_parts=2,
                           num_iterations=[1, 1],
                           batch_size=64,
                           context_size=corpus.num_words_in_window - 1)
    probes = [p for p in corpus.x if p in prep.store.w2id]

    # check that types in corpus and prep are identically ordered
    for t1, t2, in zip(prep.store.types, corpus.types):
        assert t1 == t2

    # check that each x category occurs equally often
    for cat_id in range(corpus.num_fragments):
        num = len([w for w in prep.store.tokens if w in corpus.cat_id2x[cat_id]])
        print(f'cat={cat_id+1} occurs {num:,} times')

    # get outcomes - the words that occur in the same 2-word window
    cx, ry, cx_ry = get_outcomes(prep, probes)

    # make co-occurrence matrix
    cf_mat = np.ones((corpus.num_y, corpus.num_x))
    for cxi, ryi in zip(cx, ry):
        cf_mat[corpus.y.index(ryi), corpus.x.index(cxi)] += 1

    # make co-occurrence plot
    if SHOW_HEATMAP:
        print(np.max(cf_mat))
        print(np.min(cf_mat))
        fig, ax = make_heatmap_fig(cf_mat)
        ce = drv.entropy_conditional(cx, ry).item()
        je = drv.entropy_joint(cx_ry).item()
        ye = drv.entropy_joint(ry).item()
        plt.title(f'Toy Corpus\nnum sentinels={ns}\nH(x-word|y-word)={ce:.4f}\nH(x-word,y-word)={je:.4f}\nH(y-word)={ye:.4f}')
        plt.show()

    # collect singular values for plotting
    cf_mat_intact = scale(cf_mat, axis=1, with_std=False, with_mean=False)
    cf_mat_scaled = scale(cf_mat, axis=1, with_std=False, with_mean=True)  # subtracting mean from rows
    s_intact = np.linalg.svd(cf_mat_intact, compute_uv=False)
    s_scaled = np.linalg.svd(cf_mat_scaled, compute_uv=False)
    s_list_intact.append(np.asarray(s_intact[:NUM_S_DIMS]))
    s_list_scaled.append(np.asarray(s_scaled[:NUM_S_DIMS]))

# difference between normalizing and no normalizing matters!
plot_singular_values(s_list_intact, scaled=bool(0), max_s=NUM_S_DIMS, label_name='ns', label_values=NUM_SENTINELS_LIST)
plot_singular_values(s_list_scaled, scaled=bool(1), max_s=NUM_S_DIMS, label_name='ns', label_values=NUM_SENTINELS_LIST)

