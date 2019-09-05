

class Eval:
    num_opt_init_steps = 0
    num_opt_steps = 5
    xi = 0.01  # 0.01 is better than 0.05
    verbose = True
    eval_thresholds = [[0.9999], [0.999], [0.99]]


class Figs:
    title_label_fs = 8
    axis_fs = 12
    leg_fs = 12