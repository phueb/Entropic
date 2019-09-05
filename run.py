import numpy as np
from scipy.stats import t

from src.experiment import simulate
from src.plot import plot_trajectories
from src.utils import to_eval_epochs
from src import config


class Opts:

    scale_weights = 1.0  # works with 1.0 but not with 0.01 or 0.1
    lr = 1.0  # TODO vary
    hidden_size = 8
    num_epochs =10 * 1000

    separate_feedback = [False, 0.5]  # probability of using only subordinate feedback for a single item
    y2_noise = [False, 0.1]  # probability of switching the superordinate label for a single item

    num_reps = 5
    num_evals = 10
    representation = 'output'

    num_subordinate_cats_in_a = 3
    num_subordinate_cats_in_b = 3
    subordinate_size = 3


"""
structure of the data:
there are 2 superordinate categories (category A and B).
each superordinate category contains NUM_SUBORDINATES subordinate categories.
each subordinate category contains SUBORDINATE_SIZE items.
"""

eval_epochs = to_eval_epochs(Opts)
print('evaluating at epochs:')
print(eval_epochs)

# get simulation results - returns tensor with shape = [NUM_REPS, 2, Opt.num_evals]
init_conditions = ['identical', 'linear', 'random']
results_1 = np.array([simulate(init_conditions[0], Opts) for _ in range(Opts.num_reps)])
results_2 = np.array([simulate(init_conditions[1], Opts) for _ in range(Opts.num_reps)])
results_3 = np.array([simulate(init_conditions[2], Opts) for _ in range(Opts.num_reps)])

# plot results
if config.Eval.score_a and config.Eval.score_b:
    cat_names = ['A', 'B']
elif not config.Eval.score_a:
    cat_names = ['B']
elif not config.Eval.score_b:
    cat_names = ['A']
else:
    raise SystemExit('config.Eval.score_a and config.Eval.score_b are set to False')
#
for n, cat_name in enumerate(cat_names):
    # aggregate data
    xs = [eval_epochs] * len(init_conditions)
    ys = [results_1.mean(axis=0, keepdims=True)[:, n, :].squeeze(),
          results_2.mean(axis=0, keepdims=True)[:, n, :].squeeze(),
          results_3.mean(axis=0, keepdims=True)[:, n, :].squeeze()]
    stds = [results_1.std(axis=0, keepdims=True)[:, n, :].squeeze(),
            results_2.std(axis=0, keepdims=True)[:, n, :].squeeze(),
            results_3.std(axis=0, keepdims=True)[:, n, :].squeeze()]
    sems = [std / np.sqrt(Opts.num_reps) for std in stds]
    margins_of_error = [sem * t.ppf(1 - 0.05 / 2, Opts.num_reps - 1) for sem in sems]  # margins are 1/2 the total CI
    options = '\n'.join(['{}={}'.format(k, v) for k, v in sorted(Opts.__dict__.items())
                         if not k.startswith('_')])
    # fig
    fig = plot_trajectories(xs=xs,
                            ys=ys,
                            margins_of_error=margins_of_error,
                            options=options,
                            labels=init_conditions,
                            label_prefix='init = ',
                            name='category {} '.format(cat_name) + config.Eval.metric,
                            ylim=[0.5, 1.0] if config.Eval.metric in ['ba', 'fs'] else [0.0, 1.0],
                            figsize=(6, 6))
    fig.show()
