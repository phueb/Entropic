"""
Note:
To convert images into a gif, in terminal:
convert -delay 5 FOLDER_NAME/*.png test.gif
"""

import numpy as np
import yaml
import shutil

from entropic import configs
from entropic.params import param2default, param2requests
from entropic.figs import make_predictions_animation

from ludwig.results import gen_param_paths


SLOT = 'b'
LABEL_PARAMS = []  # any additional params to put into label



def to_step(file_name):
    return int(''.join([i for i in file_name if i.isdigit()]))


summary_data = []
for param_path, label in gen_param_paths(configs.Dirs.root.name,
                                         param2requests,
                                         param2default,
                                         label_n=False,
                                         label_params=LABEL_PARAMS):
    # num_jobs
    job_paths = list(param_path.glob(f'*num*'))
    num_jobs = len(job_paths)

    # num_ticks
    npy_paths = list(param_path.glob(f'*num*/saves/output_probabilities_{SLOT}_*.npy'))
    if not npy_paths:
        raise SystemExit('Did not find any .npy files')
    step_set = set([to_step(p.name) for p in npy_paths])
    num_ticks = len(step_set)

    # num_fragments, num_types
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    num_fragments = param2val['num_fragments']
    num_types = param2val['num_types']
    hidden_size = param2val['hidden_size']

    # init array to hold category representations averaged across jobs
    big = np.zeros((num_jobs, num_ticks, num_fragments, num_types))

    # calc steps_in_tick
    max_step = max(step_set)
    steps_in_tick = max_step / (num_ticks - 1)  # -1 to account for tick at step=0
    assert steps_in_tick.is_integer()
    steps_in_tick = int(steps_in_tick)

    # get a 3d array for each job and collect in 4d array
    for job_id, job_path in enumerate(job_paths):

        # collect all representations from one job into 4d array
        for npy_path in sorted(job_path.glob(f'saves/output_probabilities_{SLOT}_*.npy')):
            r_job = np.load(npy_path)
            tick = to_step(npy_path.name) // steps_in_tick
            print(f'Reading {npy_path.name} tick={tick:>3}')
            # make category representations (averages over same-category word representations)
            cat_reps = np.vstack([r_job[offset::num_fragments].mean(0) for offset in range(num_fragments)])
            big[job_id, tick] += cat_reps

        # make path where to save images
        label_flat = label.replace('\n', '-')
        images_path = configs.Dirs.images / f'{label_flat}_{job_id:0>3}_{SLOT}'
        if not images_path.exists():
            images_path.mkdir()
        else:
            shutil.rmtree(str(images_path))
            images_path.mkdir()

        # get tick at which delay occurs
        delay_tick = 0  # todo no longer used

        make_predictions_animation(big[job_id],
                                   label=label,
                                   slot=SLOT,
                                   steps_in_tick=steps_in_tick,
                                   delay_tick=delay_tick,  # tick at which delay occurs
                                   num_fragments=num_fragments,
                                   images_path=images_path,
                                   )

