from typing import Dict, Optional

import pandas as pd
from scipy import stats

from entropic.figs import plot_summary, correct_artifacts
from entropic import config
from entropic.params import param2default, param2requests

from ludwig.results import gen_param_paths


LABEL_PARAMS = []  # must be a list
SLOTS = ['a']
CONTEXT_SIZE = 1
LEGEND = True
LABELS = []
TOLERANCE = 0.04
SORT_BY_PERFORMANCE = True

STUDY = '0a2'  # '1a'
SIMPLIFY_STUDY2 = True


if STUDY == '0a1':
    param2requests = {'sample_a': [('item', 'item')],
                      'num_sentinels': [8],
                      'incongruent_a': [(0.4, 0.0), (0.0, 0.4)],
                      }
elif STUDY == '0a2':
    param2requests = {'sample_a': [('super', 'super')],
                      'num_sentinels': [8],
                      'size_a': [(1.1, 0.1), (0.1, 1.1), (1.1, 1.1)],
                      }

if STUDY == '1a':
    param2requests = {'sample_a': [('super', 'super'), ('sub', 'sub'), ('item', 'item')],
                      'num_sentinels': [4],
                      }
elif STUDY == '1b':
    param2requests = {'sample_b': [('super', 'super'), ('sub', 'sub'), ('item', 'item')],
                      'num_sentinels': [4],
                      }

elif STUDY == '2a':
    param2requests = {'sample_a': [('super', 'item'), ('super', 'super'), ('item', 'item'), ('item', 'super')],
                      'num_sentinels': [4],
                      }
elif STUDY == '2b':
    param2requests = {'sample_b': [('super', 'item'), ('super', 'super'),  ('item', 'item'), ('item', 'super')],
                      'num_sentinels': [4],
                      }

elif STUDY == '3a1':
    param2requests = {'sample_a': [('super', 'item'), ('item', 'item')],
                      'incongruent_a': [(0.0, 0.1), (0.0, 0.0)],
                      'num_sentinels': [4],
                      }
elif STUDY == '3a2':
    param2requests = {'sample_a': [('super', 'item'), ('item', 'item')],
                      'incongruent_a': [(0.1, 0.1), (0.0, 0.0)],
                      'num_sentinels': [4],
                      }
elif STUDY == '3a3':
    param2requests = {'sample_a': [('super', 'item'), ('item', 'super')],
                      'incongruent_a': [(0.1, 0.1), (0.0, 0.0)],
                      'num_sentinels': [4],
                      }
elif STUDY == '3a4':
    param2requests = {'sample_a': [('item', 'item')],
                      'incongruent_a': [(0.1, 0.0), (0.0, 0.1), (0.1, 0.1), (0.0, 0.0)],
                      'num_sentinels': [4],
                      }


if SIMPLIFY_STUDY2 and str(STUDY).startswith('2'):  # show only most important results
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
                                                 label_n=True,
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
        label = label.replace(f'num_sentinels={param2requests["num_sentinels"][0]}\n', '')

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
                       title=f'study={STUDY}\nslot={slot}\ncontext-size={CONTEXT_SIZE}\n',
                       )
    fig.show()
