from collections import Counter
from typing import Union
import numpy as np
from bayes_opt import BayesianOptimization
from functools import partial

from preppy import PartitionedPrep, SlidingPrep

from entropic import config
from entropic.outcomes import get_outcomes


def calc_cluster_score(sim_mat, gold_mat, cluster_metric):
    print('Computing {} score...'.format(cluster_metric))

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

    def calc_probes_fs(thr):
        """
        WARNING: this gives incorrect results at early timepoints (lower compared to tensorflow implementation)
        # TODO this not due to using sim_mean as first point to bayesian-opt:
        # TODO perhaps exploration-exploitation settings are only good for ba but not f1

        """
        tp, tn, fp, fn = calc_signals_partial(thr)
        precision = np.divide(tp + 1e-7, (tp + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        fs = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-7)
        return fs

    def calc_probes_ck(thr):
        """
        cohen's kappa
        """
        tp, tn, fp, fn = calc_signals_partial(thr)
        totA = np.divide(tp + tn, (tp + tn + fp + fn))
        #
        pyes = ((tp + fp) / (tp + fp + tn + fn)) * ((tp + fn) / (tp + fp + tn + fn))
        pno = ((fn + tn) / (tp + fp + tn + fn)) * ((fp + tn) / (tp + fp + tn + fn))
        #
        randA = pyes + pno
        ck = (totA - randA) / (1 - randA)
        # print('totA={:.2f} randA={:.2f}'.format(totA, randA))
        return ck

    def calc_probes_ba(thr):
        tp, tn, fp, fn = calc_signals_partial(thr)
        specificity = np.divide(tn + 1e-7, (tn + fp + 1e-7))
        sensitivity = np.divide(tp + 1e-7, (tp + fn + 1e-7))  # aka recall
        ba = (sensitivity + specificity) / 2  # balanced accuracy
        return ba

    # use bayes optimization to find best_thr
    gp_params = {"alpha": 1e-5, "n_restarts_optimizer": 2}  # without this, warnings about predicted variance < 0
    if cluster_metric == 'f1':
        fun = calc_probes_fs
    elif cluster_metric == 'ba':
        fun = calc_probes_ba
    elif cluster_metric == 'ck':
        fun = calc_probes_ck
    else:
        raise AttributeError('Invalid arg to "cluster_metric".')
    bo = BayesianOptimization(fun, {'thr': (0.0, 1.0)}, verbose=config.Eval.verbose)
    bo.init_points.extend(config.Eval.eval_thresholds)
    bo.maximize(init_points=config.Eval.num_opt_init_steps, n_iter=config.Eval.num_opt_steps,
                acq="poi", xi=config.Eval.xi, **gp_params)  # smaller xi: exploitation
    best_thr = bo.res['max']['max_params']['thr']
    # use best_thr
    results = fun(best_thr)
    res = np.mean(results)
    return res


def softmax(z):
    a = 1  # should be 1 if rows should sum to 1
    z_norm = np.exp(z - np.max(z, axis=a, keepdims=True))
    res = np.divide(z_norm, np.sum(z_norm, axis=a, keepdims=True))

    # check that softmax works correctly - row sum must be close to 1
    assert round(res[0, :].sum().item(), 2) == 1

    return res


def make_xw_p(prep: Union[PartitionedPrep, SlidingPrep],
              token_ids_array: np.ndarray,
              xw: str,
              ) -> np.ndarray:
    """
    make the true next-word probability distribution for some x-word
    """

    x, y, x_y = get_outcomes(prep, token_ids_array, [xw])  # outcomes where xw is in slot -2
    wid2f = Counter(y)
    res = np.asarray([wid2f[i] for i in range(prep.num_types)])
    res = res / res.sum()

    # TODO is this correct? why does dp not drop lower?

    print(f'"{xw}" occurs with {len(wid2f)} y-word types, and occurs {len(y)} times')
    assert np.sum(res).round(4).item() == 1.0, np.sum(res).round(4).item()

    return res