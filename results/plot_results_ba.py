import pandas as pd
from scipy import stats
import numpy as np

from entropic.figs import plot_summary
from entropic import config
from entropic.params import param2default, param2requests

from ludwig.results import gen_param_paths


LABEL_PARAMS = []  # must be a list
Y_LABEL = 'Categorization [Balanced accuracy]'
SLOTS = ['v', 'w', 'x', 'y']
LEGEND = True
LABELS = []
TOLERANCE = 0.04


def correct_artifacts(y: pd.Series, tolerance: float = TOLERANCE):
    """
    correct y when y drops more than tolerance.
    this is necessary because computation of balanced accuracy occasionally results in unwanted negative spikes
    """
    res = np.asarray(y)
    for i in range(len(res) - 2):
        val1, val2, val3 = res[[i, i+1, i+2]]
        if (val1 - tolerance) > val2 < (val3 - tolerance):
            res[i+1] = np.mean([val1, val3])
            print('Adjusting {} to {}'.format(val2, np.mean([val1, val3])))
        # in case dip is at end
        elif (val2 - tolerance) > val3:
            res[i+2] = val2
            print('Adjusting {} to {}'.format(val2, np.mean([val1, val3])))
    return res.tolist()


for slot in SLOTS:

    labels = iter(LABELS)

    # collect data
    summary_data = []
    for param_p, label in gen_param_paths(config.Dirs.root.name,
                                          param2requests,
                                          param2default,
                                          label_params=LABEL_PARAMS):
        # param_df
        dfs = []
        for df_p in param_p.glob(f'*num*/ba_{slot}.csv'):
            print('Reading {}'.format(df_p.name))
            df = pd.read_csv(df_p, index_col=0)
            df.index.name = 'step'
            # remove dips
            df = df.apply(correct_artifacts, result_type='expand')
            dfs.append(df)
        param_df = frame = pd.concat(dfs, axis=1)

        # custom labels
        if LABELS:
            label = next(labels)

        # summarize data
        num_reps = param_df.shape[1]
        summary_data.append((param_df.index.values,
                             param_df.mean(axis=1).values,
                             param_df.sem(axis=1).values * stats.t.ppf(1 - 0.05 / 2, num_reps - 1),
                             label,
                             num_reps))
        print('--------------------- END {}\n\n'.format(param_p.name))

    # sort data
    summary_data = sorted(summary_data, key=lambda data: sum(data[1]), reverse=True)
    if not summary_data:
        raise SystemExit('No data found')

    # plot
    fig = plot_summary(summary_data,
                       y_label=Y_LABEL,
                       legend=LEGEND,
                       title=f'slot={slot}'
                       )
    fig.show()
