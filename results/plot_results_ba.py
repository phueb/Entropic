"""
Note:
    balanced accuracy is flat when evaluating slot=b and context-size=3 and sample_b=super,
    because items in slot B do not correlate with items in slot Y (sampling strategy is
    "super"). The context does not have category information because for each Bi word,
     all category contexts are equally represented, thus averaging out category information.
     If instead, context information for each Yi item was averaged together, and Yi item information discarded,
     balanced accuracy should not be flat. This is not done because the research question is about category information
     in A, X, and B, and not in Y.


"""

import pandas as pd
from scipy import stats
import shutil

from entropic.figs import plot_summary, correct_artifacts
from entropic import configs
from entropic.params import param2default, param2requests

from ludwig.results import gen_param_paths


LABEL_PARAMS = []  # must be a list
LEGEND = True
LABELS = []
TOLERANCE = 0.04
SORT_BY_PERFORMANCE = True
REVERSE_ORDER = True

LSTM = False
STUDY = '22a'


if STUDY == '11a':
    param2requests = {
        'redundant_a': [((0.0, 0.0), (0.0, 0.0)),
                        ((0.5, 0.5), (0.5, 0.5)),
                        ((1.0, 1.0), (1.0, 1.0))],
    }
    conditions = [('a', 1), ('x', 1)]
elif STUDY == '11b':
    param2requests = {
        'redundant_b': [((0.0, 0.0), (0.0, 0.0)),
                        ((0.5, 0.5), (0.5, 0.5)),
                        ((1.0, 1.0), (1.0, 1.0))],
    }
    conditions = [('x', 1), ('b', 1)]


elif STUDY == '21a':
    param2requests = {
        'redundant_a': [
            ((0.0, 0.0), (1.0, 1.0)),
            ((0.5, 0.5), (1.0, 1.0)),
            # ((0.9, 0.9), (1.0, 1.0)),
            ((1.0, 1.0), (1.0, 1.0))],
    }
    conditions = [('a', 1), ('x', 1)]
elif STUDY == '21b':
    param2requests = {
        'redundant_b': [
            ((0.0, 0.0), (1.0, 1.0)),
            ((0.5, 0.5), (1.0, 1.0)),
            ((1.0, 1.0), (1.0, 1.0))],
    }
    conditions = [('x', 1), ('b', 1)]

elif STUDY == '22a':
    param2requests = {
        'redundant_a': [
            ((0.0, 0.0), (1.0, 1.0)),
            ((1.0, 1.0), (0.0, 0.0))],
    }
    conditions = [('a', 1), ('x', 1)]
elif STUDY == '22b':
    param2requests = {
        'redundant_b': [
            ((0.0, 0.0), (1.0, 1.0)),
            ((1.0, 1.0), (0.0, 0.0))],
    }
    conditions = [('x', 1), ('b', 1)]

else:
    conditions = [('a', 1), ('x', 1), ('b', 1)]


if LSTM:  # need to change both flavor, and lr
    param2requests['flavor'] = ['lstm']
    param2requests['lr'] = [1.0]

if configs.Dirs.summaries.exists():
    shutil.rmtree(configs.Dirs.summaries)

for slot, context_size in conditions:

    labels = iter(LABELS)

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
        for df_p in param_p.glob(f'*num*/ba_{slot}_context-size={context_size}.csv'):
            print('Reading {}'.format(df_p.name))
            df = pd.read_csv(df_p, index_col=0)
            df.index.name = 'step'
            # remove dips
            df = df.apply(correct_artifacts, result_type='expand', tolerance=TOLERANCE)
            dfs.append(df)
        param_df: pd.DataFrame = pd.concat(dfs, axis=1)

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

        # export to csv - latex does not like spaces in file name
        fn = label.replace("\n", "-").replace(' ', '') + f'-slot={slot}-cs={context_size}'
        path = configs.Dirs.summaries / f'{fn}.txt'  # txt format makes file content visible on overleaf.org
        if not path.parent.exists():
            path.parent.mkdir()
        exported_df = pd.DataFrame(data={'mean': param_df.mean(axis=1).values,
                                         'std': param_df.std(axis=1).values},
                                   index=param_df.index.values)
        exported_df.index.name = 'step'
        exported_df.round(3).to_csv(path, sep=' ')

    if not summary_data:
        raise SystemExit('No data found')

    # sort data
    if SORT_BY_PERFORMANCE:
        summary_data = sorted(summary_data, key=lambda data: sum(data[1]), reverse=True)

    # plot
    v_line = summary_data[0][0][-1] / 2
    fig = plot_summary(summary_data,
                       y_label='Categorization [Balanced accuracy]',
                       legend=LEGEND,
                       v_line=v_line,
                       title=f'study={STUDY}\nslot={slot}\ncontext-size={context_size}\n',
                       )
    fig.show()

