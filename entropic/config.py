from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = root / 'entropic'
    images = root / 'images'


class Eval:
    num_opt_init_steps = 5
    num_opt_steps = 10
    xi = 0.01  # 0.01 is better than 0.05
    verbose = False
    eval_thresholds = [[0.999], [0.99], [0.9], [0.2], [0.01], [0.001]]
    eval_interval = 10
    calc_dp = False
    save_output_probabilities = False
    save_embeddings = False


class Model:
    max_w = None  # best to leave this as None, any higher makes the prior of the RNN less entropic


class Fig:
    title_label_fs = 8
    axis_fs = 12
    leg_fs = 8
    fig_size = (6, 6)
    dpi = 163
    line_width = 2
