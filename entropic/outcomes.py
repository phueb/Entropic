import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import List, Tuple

from preppy import PartitionedPrep


def get_outcomes(prep: PartitionedPrep,
                 probes: List[str],
                 ) -> Tuple[List[str], List[str], np.ndarray]:

    # windows
    num_possible_windows = len(prep.token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(prep.token_ids_array, shape, strides=(8, 8), writeable=False)

    # windows with probe in position -2
    row_ids = np.isin(windows[:, -2], [prep.store.w2id[w] for w in probes])
    probe_windows = windows[row_ids]

    # outcomes
    cx = [prep.store.types[i] for i in probe_windows[:, -2]]  # intersection of all words in center with x words
    ry = [prep.store.types[i] for i in probe_windows[:, -1]]  # intersection of all words in right with y words
    cx_ry = np.vstack((cx, ry))  # joint outcome hast to be array to play nicely with drv.entropy_joint

    return cx, ry, cx_ry