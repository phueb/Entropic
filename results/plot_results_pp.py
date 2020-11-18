import pandas as pd
from scipy import stats

from entropic.figs import plot_summary
from entropic import configs
from entropic.params import param2default, param2requests

from ludwig.results import gen_param_paths


LABEL_PARAMS = []  # must be a list
ADDITIONAL_TITLE = ''
LEGEND = True
LABELS = []
SORT_BY_PERFORMANCE = True
REVERSE_ORDER = True

labels = iter(LABELS)

param2requests = {
    'redundant_a': [(0.3, 0.3), (0.4, 0.4), (0.5, 0.5),
                    (0.6, 0.6), (0.9,0.9)],
}

# collect data
summary_data = []
color_id = 0
for param_p, label in sorted(gen_param_paths(configs.Dirs.root.name,
                                             param2requests,
                                             param2default,
                                             label_n=True,
                                             label_params=LABEL_PARAMS),
                             key=lambda i: i[1], reverse=REVERSE_ORDER):
    # load data-frame
    dfs = []
    for df_p in param_p.glob(f'*num*/pp.csv'):
        print('Reading {}'.format(df_p.name))
        df = pd.read_csv(df_p, index_col=0)
        df.index.name = 'step'
        dfs.append(df)
    param_df = frame = pd.concat(dfs, axis=1)

    # custom labels
    if LABELS:
        label = next(labels)

    # color
    color = f'C{color_id}'  # make colors consistent with label, not best ba
    if color_id < 10:  # only 10 colors in default color cycle
        color_id += 1
    else:
        color = '0'

    # summarize data
    num_reps = param_df.shape[1]
    summary_data.append((param_df.index.values,
                         param_df.mean(axis=1).values,
                         param_df.sem(axis=1).values * stats.t.ppf(1 - 0.05 / 2, num_reps - 1),
                         label,
                         color,
                         num_reps))

if not summary_data:
    raise SystemExit('No data found')

# sort data
if SORT_BY_PERFORMANCE:
    summary_data = sorted(summary_data, key=lambda data: sum(data[1]), reverse=True)

# plot
fig = plot_summary(summary_data,
                   y_label='Prediction Performance [Perplexity]',
                   legend=LEGEND,
                   title=f'{ADDITIONAL_TITLE}',
                   y_lims=(0., 40.),
                   )
fig.show()
