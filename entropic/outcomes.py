import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import List, Tuple

from preppy import PartitionedPrep


def get_outcomes(prep: PartitionedPrep,
                 token_ids_array: np.ndarray,
                 probes: List[str],
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # windows
    num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)

    # windows with probe in position -2
    row_ids = np.isin(windows[:, -2], [prep.store.w2id[w] for w in probes])
    probe_windows = windows[row_ids]

    # outcomes
    x = probe_windows[:, -2]  # x-word
    y = probe_windows[:, -1]  # y-word
    x_y = np.vstack((x, y))  # joint outcome

    return x, y, x_y