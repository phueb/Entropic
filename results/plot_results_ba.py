import pandas as pd
from scipy import stats

from entropic.figs import plot_summary, correct_artifacts
from entropic import config
from entropic.params import param2default, param2requests

from ludwig.results import gen_param_paths


LABEL_PARAMS = []  # must be a list
ADDITIONAL_TITLE = ''
SLOTS = ['x']
CONTEXT_SIZE = 2
LEGEND = True
LABELS = []
TOLERANCE = 0.04

STUDY = '1a'  # '1a'


if STUDY == '1a':
    del param2requests
    param2requests = {'sample_a': [('super', 'super'), ('sub', 'sub'), ('item', 'item')],
                      }
elif STUDY == '1b':
    del param2requests
    param2requests = {'sample_b': [('super', 'super'), ('sub', 'sub'), ('item', 'item')],
                      }
elif STUDY == '2a':
    del param2requests
    param2requests = {'sample_a': [('super', 'item'), ('super', 'super'), ('item', 'item'), ('item', 'super')],
                      }
elif STUDY == '2b':
    del param2requests
    param2requests = {'sample_b': [('super', 'item'), ('super', 'super'),  ('item', 'item'), ('item', 'super')],
                      }


for slot in SLOTS:

    labels = iter(LABELS)

    # collect data
    summary_data = []
    color_id = 0
    for param_p, label in gen_param_paths(config.Dirs.root.name,
                                          param2requests,
                                          param2default,
                                          label_params=LABEL_PARAMS):
        # param_df
        dfs = []
        for df_p in param_p.glob(f'*num*/ba_{slot}_context-size={CONTEXT_SIZE}.csv'):
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

        label = label.replace('sample_', '')

        color = f'C{color_id}'  # make colors consistent with label, not best ba
        color_id += 1

        # summarize data
        num_reps = param_df.shape[1]
        summary_data.append((param_df.index.values,
                             param_df.mean(axis=1).values,
                             param_df.sem(axis=1).values * stats.t.ppf(1 - 0.05 / 2, num_reps - 1),
                             label,
                             color,
                             num_reps))

    # sort data
    summary_data = sorted(summary_data, key=lambda data: sum(data[1]), reverse=True)
    if not summary_data:
        raise SystemExit('No data found')

    # plot
    fig = plot_summary(summary_data,
                       y_label='Categorization [Balanced accuracy]',
                       legend=LEGEND,
                       title=f'slot={slot}\ncontext-size={CONTEXT_SIZE}\n{ADDITIONAL_TITLE}',
                       )
    fig.show()
