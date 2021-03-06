from collections import Counter
from typing import Union, List, Tuple
import numpy as np
from bayes_opt import BayesianOptimization
from functools import partial
import warnings

from preppy import PartitionedPrep, SlidingPrep

from entropic import configs


def calc_ba(sim_mat, gold_mat):
    warnings.filterwarnings('ignore')

    def calc_signals(_sim_mat, _labels, thr):  # vectorized algorithm is 20X faster
        probe_sims_clipped = np.clip(_sim_mat, 0, 1)
        probe_sims_clipped_triu = probe_sims_clipped[np.triu_indices(len(probe_sims_clipped), k=1)]
        predictions = np.zeros_like(probe_sims_clipped_triu, int)
        predictions[np.where(probe_sims_clipped_triu > thr)] = 1
        #
        tp = float(len(np.where((predictions == _labels) & (_labels == 1))[0]))
        tn = float(len(np.where((predictions == _labels) & (_labels == 0))[0]))
        fp = float(len(np.where((predictions != _labels) & (_labels == 0))[0]))
        fn = float(len(np.where((predictions != _labels) & (_labels == 1))[0]))
        return tp, tn, fp, fn

    # define calc_signals_partial
    labels = gold_mat[np.triu_indices(len(gold_mat), k=1)]
    calc_signals_partial = partial(calc_signals, sim_mat, labels)

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals_partial(thr)
        specificity = np.divide(tn + 1e-7, (tn + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}  # without this, warnings about predicted variance < 0
    bo = BayesianOptimization(calc_probes_ba, {'thr': (0.0, 1.0)}, verbose=configs.Eval.verbose)
    bo.init_points.extend(configs.Eval.eval_thresholds)
    bo.maximize(init_points=configs.Eval.num_opt_init_steps, n_iter=configs.Eval.num_opt_steps,
                acq="poi", xi=configs.Eval.xi, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = calc_probes_ba(best_thr)
    res = np.mean(results)
    return res


def softmax(z):
    a = 1  # should be 1 if rows should sum to 1
    z_norm = np.exp(z - np.max(z, axis=a, keepdims=True))
    res = np.divide(z_norm, np.sum(z_norm, axis=a, keepdims=True))

    # check that softmax works correctly - row sum must be close to 1
    assert round(res[0, :].sum().item(), 2) == 1

    return res


def make_p_cat(prep: Union[PartitionedPrep, SlidingPrep],
               x: List[str],
               types: List[str],
               ) -> np.ndarray:
    """
    make the true next-word probability distribution for some x-word
    """
    x_windows = get_windows(prep, x, col_id=-2)  # windows with x in slot -2
    cx, ry, cx_ry = get_outcomes(prep, x_windows)
    w2f = Counter(ry)
    res = np.asarray([w2f[w] for w in types])
    res = res / res.sum()

    print(f'Provided x-word fragment occurs with {len(w2f)} y-word types, and occurs {len(ry)} times')
    assert np.sum(res).round(4).item() == 1.0, np.sum(res).round(4).item()

    return res


def get_outcomes(prep: Union[PartitionedPrep, SlidingPrep],
                 windows: np.ndarray,
                 ) -> Tuple[List[str], List[str], np.ndarray]:

    # outcomes
    cx = [prep.store.types[i] for i in windows[:, -3]]  # intersection of all words in center with x words
    ry = [prep.store.types[i] for i in windows[:, -1]]  # intersection of all words in right with y words
    cx_ry = np.vstack((cx, ry))  # joint outcome hast to be array to play nicely with drv.entropy_joint

    return cx, ry, cx_ry


def get_windows(prep: Union[PartitionedPrep, SlidingPrep],
                words: List[str],
                col_id: int,
                ) -> np.ndarray:
    row_ids = np.isin(prep.reordered_windows[:, col_id], [prep.store.w2id[w] for w in words])
    res = prep.reordered_windows[row_ids]
    return res
