import matplotlib.pyplot as plt

from src import config


def plot_trajectories(xs, ys, labels, label_prefix, name, figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('epoch', fontsize=config.Figs.axlabel_fs)
    ax.set_ylabel(name + '\n+/- std dev', fontsize=config.Figs.axlabel_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    #
    for x, y, label in zip(xs, ys, labels):
        ax.plot(x, y, label=label_prefix + str(label))

    plt.legend(fontsize=config.Figs.leg_fs, frameon=False, loc='lower right', ncol=1)
    return fig