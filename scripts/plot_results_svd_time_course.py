import numpy as np
import yaml
import shutil

from entropic import config
from entropic.params import param2default, param2requests
from entropic.figs import make_svd_across_time_fig
from entropic.figs import make_svd_across_time_3d_animation
from entropic.figs import make_svd_across_time_3d_fig

from ludwig.results import gen_param_paths


LABEL_TICK_INTERVAL = 10
PLOT_INDIVIDUAL_STATIC_FIGURE = False
PLOT_INDIVIDUAL_ANIMATION = True
REPRESENTATIONS_NAME = 'output_probabilities'  # "embeddings' or "output_probabilities"


# param2requests['hidden_size'] = [32]

"""
Note:
To convert images into a gif, in terminal:
convert -delay 5 FOLDER_NAME/*.png test.gif
"""


def to_step(file_name):
    return int(''.join([i for i in file_name if i.isdigit()]))


summary_data = []
for param_path, label in gen_param_paths(config.Dirs.root.name,
                                         param2requests,
                                         param2default,
                                         label_n=False,
                                         label_params=['period_probability']):
    # num_jobs
    job_paths = list(param_path.glob(f'*num*'))
    num_jobs = len(job_paths)

    # num_ticks
    npy_paths = list(param_path.glob(f'*num*/saves/{REPRESENTATIONS_NAME}_*.npy'))
    step_set = set([to_step(p.name) for p in npy_paths])
    num_ticks = len(step_set)

    # num_fragments, num_types
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    num_fragments = param2val['num_fragments']
    num_types = param2val['num_types']
    hidden_size = param2val['hidden_size']

    if REPRESENTATIONS_NAME == 'output_probabilities':
        representation_size = num_types
    elif REPRESENTATIONS_NAME == 'embeddings':
        representation_size = hidden_size
    else:
        raise AttributeError('Invalid arg to REPRESENTATIONS_NAME.')

    # init array to hold category representations averaged across jobs
    big = np.zeros((num_jobs, num_ticks, num_fragments, representation_size))

    # calc steps_in_tick
    max_step = max(step_set)
    steps_in_tick = max_step / (num_ticks - 1)  # -1 to account for tick at step=0
    assert steps_in_tick.is_integer()
    steps_in_tick = int(steps_in_tick)

    # get a 3d array for each job and collect in 4d array
    for job_id, job_path in enumerate(job_paths):

        # load collect all representations from one job into 4d array
        for npy_path in sorted(job_path.glob(f'saves/{REPRESENTATIONS_NAME}_*.npy')):
            r_job = np.load(npy_path)
            tick = to_step(npy_path.name) // steps_in_tick
            print(f'Reading {npy_path.name} tick={tick:>3}')
            # make category representations (averages over same-category word representations)
            cat_reps = np.vstack([r_job[offset::num_fragments].mean(0) for offset in range(num_fragments)])
            big[job_id, tick] += cat_reps

        if PLOT_INDIVIDUAL_STATIC_FIGURE:
            fig = make_svd_across_time_3d_fig(big[job_id],
                                              component1=0,
                                              component2=1,
                                              component3=2,
                                              label='Time course for a single simulation',
                                              label_tick_interval=LABEL_TICK_INTERVAL,
                                              steps_in_tick=steps_in_tick)
            fig.show()

        if PLOT_INDIVIDUAL_ANIMATION:
            label_flat = label.replace('\n', '-')
            images_path = config.Dirs.images / f'{label_flat}_{job_id:0>3}'
            if not images_path.exists():
                images_path.mkdir()
            else:
                shutil.rmtree(str(images_path))
                images_path.mkdir()
            make_svd_across_time_3d_animation(big[job_id],
                                              component1=0,
                                              component2=1,
                                              component3=2,
                                              label=label,
                                              steps_in_tick=steps_in_tick,
                                              images_path=images_path,
                                              )

    # get average
    rep_time_course_avg = np.sum(big, axis=0) / num_jobs

    # plot in 3d
    if num_fragments == 4:
        fig = make_svd_across_time_3d_fig(rep_time_course_avg,
                                          component1=0,
                                          component2=1,
                                          component3=2,
                                          label='Average over jobs\n' + label,
                                          label_tick_interval=LABEL_TICK_INTERVAL,
                                          steps_in_tick=steps_in_tick)
    # plot in 2d
    elif num_fragments == 2:
        fig = make_svd_across_time_fig(rep_time_course_avg,
                                       component1=0,
                                       component2=1,
                                       label='Average over jobs\n' + label,
                                       steps_in_tick=steps_in_tick)
    else:
        raise AttributeError('"num_fragments" is too large to plot')

    fig.show()
