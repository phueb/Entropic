import pandas as pd
import yaml
from scipy import stats

from init_experiments.plot import plot_trajectories
from init_experiments import config
from init_experiments.params import DefaultParams as MatchParams

from ludwigcluster.utils import list_all_param2vals

LOCAL = False

CAT = '*'  # A, B or *
X_LIMS = [[0, 100], [0, 1000], [0, 5000]]  # zoom in on particular vertical region of plot


default_dict = MatchParams.__dict__.copy()
default_dict['init'] = 'setting this to a random string ensures that it shows up in legend'

MatchParams.init = ['random', 'identical']
# MatchParams.y2_noise = [[True, 0.0], [True, 0.5], [True, 1.0]]
MatchParams.scale_weights = [1.0, 10.0]


def gen_param_ps(param2requested, param2default):
    compare_params = [param for param, val in param2requested.__dict__.items()
                      if val != param2default[param]]

    runs_p = config.LocalDirs.runs.glob('*') if LOCAL else config.RemoteDirs.runs.glob('param_*')
    if LOCAL:
        print('WARNING: Looking for runs locally')

    for param_p in runs_p:
        print('Checking {}...'.format(param_p))
        with (param_p / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        param2val = param2val.copy()
        match_param2vals = list_all_param2vals(param2requested, add_names=False)
        del param2val['param_name']
        del param2val['job_name']
        if param2val in match_param2vals:
            label = '\n'.join(['{}={}'.format(param, param2val[param]) for param in compare_params])
            label += '\nn={}'.format(len(list(param_p.glob('*num*'))))
            print('Param2val matches')
            print(label)
            yield param_p, label
        else:
            print('Params do not match')


# collect data
summary_data = []
for param_p, label in gen_param_ps(MatchParams, default_dict):
    # param_df
    dfs = []
    for df_p in param_p.glob('*num*/results_{}.csv'.format(CAT)):
        print('Reading {}'.format(df_p.name))
        df = pd.read_csv(df_p, index_col=0)
        df.index.name = 'epoch'
        dfs.append(df)
    param_df = frame = pd.concat(dfs, axis=1)
    print(param_df)
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
for xlim in X_LIMS:
    fig = plot_trajectories(summary_data,
                            y_label=config.Eval.metric,  # is averaged over cat A and B
                            xlim=xlim,
                            ylim=[0.5, 1.0] if config.Eval.metric in ['ba', 'fs'] else [0.0, 1.0],
                            figsize=(6, 6))
    fig.show()
