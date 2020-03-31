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
SORT_BY_PERFORMANCE = True

STUDY = '3a'  # '1a'
SIMPLIFY = True


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

elif STUDY == '3a':
    del param2requests
    param2requests = {'sample_a': [('super', 'item'), ('item', 'item')],
                      'incongruent_a': [(0.0, 0.1)],
                      }

if SIMPLIFY and '2' in str(STUDY):  # show only most important results
    try:
        param2requests['sample_a'].remove(('super', 'item'))
        param2requests['sample_a'].remove(('item', 'item'))
    except (KeyError, ValueError) as e:
        print(e)
    try:
        param2requests['sample_b'].remove(('super', 'item'))
        param2requests['sample_b'].remove(('item', 'item'))
    except (KeyError, ValueError) as e:
        print(e)

for slot in SLOTS:

    labels = iter(LABELS)

    print(param2default)
    print()
    print(param2requests)

    # collect data
    summary_data = []
    color_id = 0
    for param_p, label in sorted(gen_param_paths(config.Dirs.root.name,
                                                 param2requests,
                                                 param2default,
                                                 label_params=LABEL_PARAMS)):
        # param_df
        dfs = []
        for df_p in param_p.glob(f'*num*/ba_{slot}_context-size={CONTEXT_SIZE}.csv'):
            print('Reading {}'.format(df_p.name))
            df = pd.read_csv(df_p, index_col=0)
            df.index.name = 'step'
            # remove dips
            df = df.apply(correct_artifacts, result_type='expand', tolerance=TOLERANCE)
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

    if not summary_data:
        raise SystemExit('No data found')

    # sort data
    if SORT_BY_PERFORMANCE:
        summary_data = sorted(summary_data, key=lambda data: sum(data[1]), reverse=True)

    # plot
    fig = plot_summary(summary_data,
                       y_label='Categorization [Balanced accuracy]',
                       legend=LEGEND,
                       title=f'study={STUDY}\nslot={slot}\ncontext-size={CONTEXT_SIZE}\n{ADDITIONAL_TITLE}',
                       )
    fig.show()
