from pathlib import Path


class RemoteDirs:
    root = Path('/media/research_data') / 'InitExperiments'
    runs = root / 'runs'


class LocalDirs:
    root = Path(__file__).parent.parent
    src = root / 'init_experiments'
    runs = root / '{}_runs'.format(src.name)


class Global:
    debug = False


class Eval:
    num_opt_init_steps = 0
    num_opt_steps = 5
    xi = 0.01  # 0.01 is better than 0.05
    verbose = False
    eval_thresholds = [[0.9999], [0.999], [0.99], [0.9]]
    metric = 'ba'
    start_epoch = 100
    num_evals = 10


class Figs:
    title_label_fs = 8
    axis_fs = 20
    leg_fs = 12