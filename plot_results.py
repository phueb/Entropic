import numpy as np
from scipy.stats import t
import pandas as pd
import yaml

from init_experiments.plot import plot_trajectories
from init_experiments import config
from init_experiments.params import DefaultParams as MatchParams

from ludwigcluster.utils import list_all_param2vals

LOCAL = False


default_dict = MatchParams.__dict__.copy()
default_dict['init'] = 'setting this to a random string ensures that it shows up in legend'

MatchParams.init = ['identical', 'random']


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
    for df_p in param_p.glob('*num*/results_*.csv'):
        print('Reading {}'.format(df_p.name))
        df = pd.read_csv(df_p, index_col=0)
        df.index.name = 'epoch'
        dfs.append(df)
    param_df = frame = pd.concat(dfs, axis=1)
    print(param_df)

    # TODO margin of error
    # sem_trajs = [std_traj / np.sqrt(Params.num_reps) for std_traj in std_trajs]
    # margins_of_error = [sem_traj * t.ppf(1 - 0.05 / 2, Params.num_reps - 1)
    # for sem_traj in sem_trajs]  # 1/2 the length CI

    # summarize data
    summary_data.append((param_df.index.values,
                         param_df.mean(axis=1).values,
                         param_df.std(axis=1).values,
                         label,
                         param_df.shape[1]))
    print('--------------------- END {}\n\n'.format(param_p.name))

# sort data
summary_data = sorted(summary_data, key=lambda data: data[1][-1], reverse=True)
if not summary_data:
    raise SystemExit('No data found')

# plot
fig = plot_trajectories(summary_data,
                        y_label=config.Eval.metric,  # is averaged over cat A and B
                        ylim=[0.5, 1.0] if config.Eval.metric in ['ba', 'fs'] else [0.0, 1.0],
                        figsize=(6, 6))
fig.show()
