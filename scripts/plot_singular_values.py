from pyitlib import discrete_random_variable as drv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

from preppy import SlidingPrep
from ludwig.results import gen_all_param2vals

from entropic.figs import make_heatmap_fig
from entropic.corpus import Corpus
from entropic.figs import plot_singular_values
from entropic.eval import get_outcomes, get_windows
from entropic.job import Params
from entropic.params import param2requests, param2default


SHOW_HEATMAP = True
NUM_S_DIMS = 8


LN = None
LVS = None  # [(0.00, 0.00)]

if LN is not None and LVS is not None:
    param2requests[LN] = LVS


s_list_scaled = []
s_list_intact = []
for param2val in gen_all_param2vals(param2requests, param2default):
    # params
    params = Params.from_param2val(param2val)
    print(params, flush=True)

    # create toy input
    corpus = Corpus(doc_size=params.num_sequences_per_doc,
                    num_types=params.num_types,
                    num_fragments=params.num_fragments,
                    starvation=params.starvation,
                    sample_b=params.sample_b,
                    sample_a=params.sample_a,
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
    # check
    assert len([p for p in corpus.x if p in prep.store.w2id]) == corpus.num_x

    # check that types in corpus and prep are identically ordered
    for t1, t2, in zip(prep.store.types, corpus.types):
        assert t1 == t2

    # check that each x category occurs equally often
    for cat_id in range(corpus.num_fragments):
        num = len([w for w in prep.store.tokens if w in corpus.cat_id2x[cat_id]])
        print(f'cat={cat_id+1} occurs {num:,} times')

    # get outcomes - the words that occur in the last 2 slots of probe windows
    x_windows = get_windows(prep, corpus.x, col_id=-3)
    cx, ry, cx_ry = get_outcomes(prep, x_windows)

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
        plt.title(f'Toy Corpus\nH(x-word|y-word)={ce:.4f}\nH(x-word,y-word)={je:.4f}\nH(y-word)={ye:.4f}')
        plt.show()

    # collect singular values for plotting
    cf_mat_intact = scale(cf_mat, axis=1, with_std=False, with_mean=False)
    cf_mat_scaled = scale(cf_mat, axis=1, with_std=False, with_mean=True)  # subtracting mean from rows
    s_intact = np.linalg.svd(cf_mat_intact, compute_uv=False)
    s_scaled = np.linalg.svd(cf_mat_scaled, compute_uv=False)
    s_list_intact.append(np.asarray(s_intact[:NUM_S_DIMS]))
    s_list_scaled.append(np.asarray(s_scaled[:NUM_S_DIMS]))

# difference between normalizing and no normalizing matters!
plot_singular_values(s_list_intact, scaled=bool(0), max_s=NUM_S_DIMS, label_name=LN, label_values=LVS)
plot_singular_values(s_list_scaled, scaled=bool(1), max_s=NUM_S_DIMS, label_name=LN, label_values=LVS)

