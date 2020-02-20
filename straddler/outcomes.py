import numpy as np
from numpy.lib.stride_tricks import as_strided
from typing import List, Tuple, Union
from collections import Counter

from preppy import PartitionedPrep, SlidingPrep


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


def make_straddler_p(prep: Union[PartitionedPrep, SlidingPrep],
                     token_ids_array: np.ndarray,
                     straddler: str,
                     ) -> np.ndarray:
    """
    make the true next-word probability distribution for the straddler word
    """

    x, y, x_y = get_outcomes(prep, token_ids_array, [straddler])  # outcomes where xw is straddler
    wid2f = Counter(y)
    res = np.asarray([wid2f[i] for i in range(prep.num_types)])
    res = res / res.sum()

    print(f'"{straddler}" occurs with {len(wid2f)} y-word types, and occurs {len(y)} times')
    assert np.sum(res).round(4).item() == 1.0, np.sum(res).round(4).item()

    return res