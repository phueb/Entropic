import matplotlib.pyplot as plt

from src import config


def plot_trajectories(xs, ys, stds, labels, label_prefix, name, figsize=(6, 6), title=''):
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title, fontsize=config.Figs.axlabel_fs)
    ax.set_xlabel('epoch', fontsize=config.Figs.axlabel_fs)
    ax.set_ylabel(name + ' +/- std dev', fontsize=config.Figs.axlabel_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    #
    colors = ['C0', 'C1', 'C2']
    for x, y, std, label, color in zip(xs, ys, stds, labels, colors):
        ax.fill_between(x, y + std, y - std, alpha=0.25, color=color)
        ax.plot(x, y, label=label_prefix + str(label), color=color)
        ax.scatter(x, y, color=color)

    plt.legend(fontsize=config.Figs.leg_fs, frameon=False, loc='lower right', ncol=1)
    return fig