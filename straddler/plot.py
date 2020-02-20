import matplotlib.pyplot as plt
import numpy as np

from straddler import config


def plot_trajectories(summary_data, y_label, ylim, xlim,
                      options='', vline=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title(options, fontsize=config.Figs.title_label_fs)
    ax.set_xlabel('epoch', fontsize=config.Figs.axis_fs)
    y_label = {'ba': 'balanced accuracy'}[y_label]
    ax.set_ylabel(y_label + '\n+/- margin of error', fontsize=config.Figs.axis_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    ax.set_ylim([ylim[0], ylim[1] + 0.05])
    ax.set_xlim(xlim)
    #
    colors = iter(['C0', 'C1', 'C2', 'C4', 'C5', 'C6'])
    for x, y, me, label, n in summary_data:
        color = next(colors)
        ax.fill_between(x, np.clip(y + me, ylim[0], ylim[1]), y - me, alpha=0.25, color=color)
        ax.plot(x, y, label=label, color=color)
        ax.scatter(x, y, color=color)
    #
    if vline is not None:
        ax.axvline(x=vline, linestyle=':', color='grey', zorder=1)
    #
    plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=1.0,
               fontsize=config.Figs.leg_fs, frameon=False, loc='upper left', ncol=1)
    plt.tight_layout()
    return fig
