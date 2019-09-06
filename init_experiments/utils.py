import numpy as np
from init_experiments import config


def to_eval_epochs(opts):
    res = [0] + list(np.rint(np.geomspace(config.Eval.start_epoch, opts.num_epochs,
                                 config.Eval.num_evals - 1)).astype(np.int))  # include 0 and last epoch
    return res