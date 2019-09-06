import numpy as np


def to_eval_epochs(opts):
    res = [0] + list(np.rint(np.logspace(0, np.log10(opts.num_epochs),
                                 opts.num_evals - 1)).astype(np.int))  # include 0 and last epoch
    return res