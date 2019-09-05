import matplotlib.pyplot as plt

from src import config


def plot_trajectories(xs, ys, margins_of_error, labels, label_prefix, name,
                      figsize=(6, 6), options='', ylim=None):
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(options, fontsize=config.Figs.title_label_fs)
    ax.set_xlabel('epoch', fontsize=config.Figs.axis_fs)
    ax.set_ylabel(name + ' +/- margin of error', fontsize=config.Figs.axis_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if ylim is not None:
        ax.set_ylim(ylim)
    #
    colors = ['C0', 'C1', 'C2']
    for x, y, me, label, color in zip(xs, ys, margins_of_error, labels, colors):
        ax.fill_between(x, y + me, y - me, alpha=0.25, color=color)
        ax.plot(x, y, label=label_prefix + str(label), color=color)
        ax.scatter(x, y, color=color)
    #
    plt.legend(fontsize=config.Figs.leg_fs, frameon=False, loc='best', ncol=1)
    return fig
