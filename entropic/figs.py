from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Tuple
import numpy as np

from entropic import config


def make_example_fig(mat,
                     xlabel='x-words',
                     ylabel='y-words'):
    fig, ax = plt.subplots(dpi=163)
    plt.title('', fontsize=5)

    # heatmap
    print('Plotting heatmap...')
    ax.imshow(mat,
              cmap=plt.get_cmap('cividis'),
              interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    # remove tick lines
    lines = (ax.xaxis.get_ticklines() +
             ax.yaxis.get_ticklines())
    plt.setp(lines, visible=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax


def add_double_legend(lines_list, labels1, labels2, y_offset=-0.3, fs=12):

    # make legend 2
    lines2 = [l[0] for l in lines_list]
    leg2 = plt.legend(lines2,
                      labels2,
                      loc='upper center',
                      bbox_to_anchor=(0.5, y_offset), ncol=2, frameon=False, fontsize=fs)

    # add legend 1
    # make legend 1 lines black but varying in style
    lines1 = [Line2D([0], [0], color='black', linestyle='-'),
              Line2D([0], [0], color='black', linestyle=':'),
              Line2D([0], [0], color='black', linestyle='--')][:len(labels1)]
    plt.legend(lines1,
               labels1,
               loc='upper center',
               bbox_to_anchor=(0.5, y_offset + 0.1), ncol=3, frameon=False, fontsize=fs)

    # add legend 2
    plt.gca().add_artist(leg2)  # order of legend creation matters here


def plot_singular_values(ys: List[np.ndarray],
                         max_s: int,
                         fontsize: int = 12,
                         figsize: Tuple[int] = (5, 5),
                         markers: bool = False,
                         label_all_x: bool = False):
    fig, ax = plt.subplots(1, figsize=figsize, dpi=None)
    plt.title('SVD of simulated co-occurrence matrix', fontsize=fontsize)
    ax.set_ylabel('Singular value', fontsize=fontsize)
    ax.set_xlabel('Singular Dimension', fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    x = np.arange(max_s) + 1  # num columns
    if label_all_x:
        ax.set_xticks(x)
        ax.set_xticklabels(x)
    # plot
    for n, y in enumerate(ys):
        ax.plot(x, y, label='toy corpus part {}'.format(n + 1), linewidth=2)
        if markers:
            ax.scatter(x, y)
    ax.legend(loc='upper right', frameon=False, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_summary(summary_data, y_label, ylim, xlim,
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