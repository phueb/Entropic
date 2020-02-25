import numpy as np
import yaml

from entropic import config
from entropic.params import param2default, param2requests
from entropic.figs import make_svd_across_time_fig

from ludwig.results import gen_param_paths


def to_step(file_name):
    return int(''.join([i for i in file_name if i.isdigit()]))


summary_data = []
for param_path, label in gen_param_paths(config.Dirs.root.name,
                                         param2requests,
                                         param2default):

    # init array to hold category representations averaged across jobs
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    num_fragments = param2val['num_fragments']
    num_types = param2val['num_types']
    npy_paths = list(param_path.glob(f'*num*/saves/output_probabilities*.npy'))
    step_set = set([to_step(p.name) for p in npy_paths])
    num_ticks = len(step_set)
    representations_avg = np.zeros((num_ticks, num_fragments, num_types))

    # calc steps_in_tick
    max_step = max(step_set)
    steps_in_tick = max_step / (num_ticks - 1)  # -1 to account for tick at step=0
    assert steps_in_tick.is_integer()
    steps_in_tick = int(steps_in_tick)

    for npy_path in sorted(npy_paths):
        # load representations for single job
        representations_job = np.load(npy_path)
        tick = to_step(npy_path.name) // steps_in_tick
        print(f'Reading {npy_path.name} tick={tick:>3}')

        # make category representations (averages over same-category word representations)
        cat_reps = np.vstack([representations_job[offset::num_fragments].mean(0) for offset in range(num_fragments)])
        representations_avg[tick] += cat_reps

    assert np.ndim(representations_avg) == 3  # (ticks, words/categories, embedding dimensions)

    # plot
    fig = make_svd_across_time_fig(representations_avg,
                                   component1=0,
                                   component2=1,
                                   label=label,
                                   steps_in_tick=steps_in_tick)
    fig.show()

    # TODO make a 3d svd time-course figure for num_fragments=4
