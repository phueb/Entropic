from matplotlib.lines import Line2D
from typing import List, Tuple, Optional
import numpy as np
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
from math import gcd

from entropic import config


def make_heatmap_fig(mat,
                     xlabel='x-words',
                     ylabel='y-words'):
    fig, ax = plt.subplots(dpi=163 * 1)
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


def equidistant_elements(l, n):
    """return "n" elements from "l" such that they are equally far apart iin "l" """
    while not gcd(n, len(l)) == n:
        l.pop()
    step = len(l) // n
    ids = np.arange(step, len(l) + step, step) - 1  # -1 for indexing
    res = np.asarray(l)[ids].tolist()
    return res


def make_svd_across_time_fig(embeddings: np.ndarray,
                             component1: int,
                             component2: int,
                             label: str,
                             num_ticks: int,
                             label_tick_interval: int = 10,
                             ) -> plt.Figure:
    """
    Returns res showing evolution of embeddings in 2D space using PCA.
    """

    assert np.ndim(embeddings) == 3  # (ticks, words/categories, embedding dimensions)

    palette = np.array(sns.color_palette("hls", embeddings.shape[1]))
    model_ticks = [n for n, _ in enumerate(embeddings)]
    equidistant_ticks = equidistant_elements(model_ticks, num_ticks)

    # fit svd model on last tick
    num_components = component2 + 1
    svd_model = TruncatedSVD(n_components=num_components)
    svd_model.fit(embeddings[-1])

    # transform embeddings at requested ticks with pca model
    transformations = []
    for ei in embeddings[equidistant_ticks]:
        transformations.append(svd_model.transform(ei)[:, [component1, component2]])

    # fig
    res, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.set_title(f'Singular dimensions {component1} and {component2}\nEvolution across training\n' + label)
    ax.axis('off')
    ax.axhline(y=0, linestyle='--', c='grey', linewidth=1.0)
    ax.axvline(x=0, linestyle='--', c='grey', linewidth=1.0)

    # plot
    for n in range(embeddings.shape[1]):

        # scatter
        x, y = zip(*[t[n] for t in transformations])
        ax.plot(x, y, c=palette[n], lw=config.Fig.line_width)

        # annotate
        for tick in range(0, len(transformations) + 1, label_tick_interval):
            x_pos, y_pos = transformations[tick][n, :]
            txt = ax.text(x_pos, y_pos, f'{tick}', fontsize=8, color=palette[n])
            txt.set_path_effects([
                patheffects.Stroke(linewidth=config.Fig.line_width, foreground="w"), patheffects.Normal()])

    return res


def plot_singular_values(ys: List[np.ndarray],
                         max_s: int,
                         pps: List[float],
                         scaled: bool,
                         fontsize: int = 12,
                         figsize: Tuple[int] = (5, 5),
                         markers: bool = False,
                         label_all_x: bool = False):
    fig, ax = plt.subplots(1, figsize=figsize, dpi=None)
    title = 'SVD of toy corpus co-occurrence matrix'
    title += f'\nrows are scaled to mean={scaled}'
    plt.title(title, fontsize=fontsize)
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
    colors = iter(['C0', 'C1', 'C3', 'C2', 'C4', 'C5', 'C6'])
    for n, y in enumerate(ys):
        color = next(colors)
        ax.plot(x, y, label=f'period prob={pps[n]}', linewidth=2, color=color)
        if markers:
            ax.scatter(x, y)
    ax.legend(loc='upper right', frameon=False, fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_summary(summary_data,
                 y_label,
                 title: str = '',
                 vline: Optional[int] = None,
                 legend: bool = True,
                 ):

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title(title, fontsize=config.Fig.title_label_fs)
    ax.set_xlabel('Training Time [step]', fontsize=config.Fig.axis_fs)
    y_label = {'ba': 'balanced accuracy',
               'dp_0_1': 'JS-Divergence [bits]\nbetween\ntrue category 1 and learned category 2 out probabilities'}[y_label]
    ax.set_ylabel(y_label + '\n+/- margin of error', fontsize=config.Fig.axis_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #
    colors = iter(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
    for x, y, me, label, n in summary_data:
        color = next(colors)
        ax.fill_between(x, y + me, y - me, alpha=0.25, color=color)
        ax.plot(x, y, label=label, color=color)
        ax.scatter(x, y, color=color)
    if vline is not None:
        ax.axvline(x=vline, linestyle=':', color='grey', zorder=1)

    if legend:
        plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=1.0,
                   fontsize=config.Fig.leg_fs, frameon=False, loc='upper left', ncol=1)

    return fig