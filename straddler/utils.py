import numpy as np
from straddler import config


def to_eval_epochs(params):
    res = [0] + list(np.rint(np.geomspace(config.Eval.start_epoch, params.num_epochs,
                                          config.Eval.num_evals - 1)).astype(np.int))  # include 0 and last epoch
    return res
