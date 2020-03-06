from mpl_toolkits import mplot3d  # this is unused but needed for 3d plotting
from matplotlib.lines import Line2D
from typing import List, Tuple, Optional, Any, Dict
import numpy as np
from sklearn.decomposition import TruncatedSVD
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
from itertools import cycle
from pathlib import Path

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


def make_svd_across_time_fig(representations: np.ndarray,
                             component1: int,
                             component2: int,
                             label: str,
                             steps_in_tick: int,
                             label_tick_interval: int = 10,
                             ) -> plt.Figure:
    """
    Returns res showing evolution of embeddings in 2D space using SVD.
    """

    assert np.ndim(representations) == 3  # (ticks, words/categories, embedding dimensions)

    palette = np.array(sns.color_palette("hls", representations.shape[1]))

    # fit svd model on last tick
    num_components = component2 + 1
    svd_model = TruncatedSVD(n_components=num_components)
    svd_model.fit(representations[-1])

    # transform embeddings at requested ticks with pca model
    transformations = []
    for ei in representations:
        transformations.append(svd_model.transform(ei)[:, [component1, component2]])

    # fig
    res, ax = plt.subplots(figsize=config.Fig.fig_size, dpi=config.Fig.dpi)
    ax.set_title(f'Singular dimensions {component1} and {component2}\nEvolution across training\n' + label)
    ax.axis('off')
    ax.axhline(y=0, linestyle='--', c='grey', linewidth=1.0)
    ax.axvline(x=0, linestyle='--', c='grey', linewidth=1.0)

    # plot
    for n in range(representations.shape[1]):

        # lines
        x, y = zip(*[t[n] for t in transformations])
        ax.plot(x, y, c=palette[n], lw=config.Fig.line_width)

        # annotate
        for tick in range(0, len(transformations) + 1, label_tick_interval):
            x_pos, y_pos = transformations[tick][n, :]
            txt = ax.text(x_pos, y_pos, f'{tick * steps_in_tick}', fontsize=8, color=palette[n])
            txt.set_path_effects([
                patheffects.Stroke(linewidth=config.Fig.line_width, foreground="w"), patheffects.Normal()])

    return res


def make_svd_across_time_3d_fig(representations: np.ndarray,
                                component1: int,
                                component2: int,
                                component3: int,
                                label: str,
                                steps_in_tick: int,
                                label_tick_interval: int = 20,
                                ) -> plt.Figure:
    """
    Returns res showing evolution of embeddings in 2D space using PCA.
    """

    assert np.ndim(representations) == 3  # (ticks, words/categories, embedding dimensions)

    assert component1 != component2
    assert component2 != component3

    palette = np.array(sns.color_palette("hls", representations.shape[1]))

    # fit svd model on last tick
    num_components = component3 + 1
    svd_model = TruncatedSVD(n_components=num_components)  # PCA gives different results
    svd_model.fit(representations[-1])

    # transform embeddings at requested ticks with svd model
    transformations = []
    for ei in representations:
        transformations.append(svd_model.transform(ei)[:, [component1, component2, component3]])

    # fig
    res = plt.figure(dpi=163 * 1)
    ax = plt.axes(projection='3d')
    ax.set_title(label)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('Singular Dim 0', labelpad=-10)
    ax.set_ylabel('Singular Dim 1', labelpad=-10)
    ax.set_zlabel('Singular Dim 2', labelpad=-10)

    # plot
    for n in range(representations.shape[1]):
        print('Plotting time-course for category', n)

        # lines
        x, y, z = zip(*[t[n] for t in transformations])
        ax.plot(x, y, z, c=palette[n], lw=config.Fig.line_width)

        # annotate
        for tick in range(0, len(transformations), label_tick_interval):
            x_pos, y_pos, z_pos = transformations[tick][n, :]
            txt = ax.text(x_pos, y_pos, z_pos, f'{tick * steps_in_tick}', fontsize=6, color=palette[n])
            txt.set_path_effects([
                patheffects.Stroke(linewidth=config.Fig.line_width, foreground="w"), patheffects.Normal()])

    return res


def make_svd_across_time_3d_animation(representations: np.ndarray,
                                      component1: int,
                                      component2: int,
                                      component3: int,
                                      label: str,
                                      steps_in_tick: int,
                                      delay_tick: int,
                                      num_sentinels: int,
                                      images_path: Path,
                                      plot_avg_location_first_three_cats: bool = False,
                                      ) -> None:
    """
    Saves figures, showing rotating 3d figure of SVD time course
    """

    assert np.ndim(representations) == 3  # (ticks, words/categories, embedding dimensions)

    assert component1 != component2
    assert component2 != component3

    num_cats = representations.shape[1]
    palette = np.array(sns.color_palette("hls", num_cats))

    # fit svd model on last tick
    # fitting on last tick is problematic when catastrophic interference occurs,
    # because in that case the singular dimensions don't remain constant,
    # so that singular dimensions at last tick do not correspond to those before catastrophic interference
    num_components = component3 + 1
    svd_model = TruncatedSVD(n_components=num_components)  # PCA gives different results
    svd_model.fit(representations[delay_tick - 2])

    # transform embeddings at requested ticks with svd model
    transformations = []
    for ei in representations:
        transformations.append(svd_model.transform(ei)[:, [component1, component2, component3]])

    # fig
    plt.figure(dpi=163)
    ax = plt.axes(projection='3d')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('Singular Dim 0', labelpad=-10)
    ax.set_ylabel('Singular Dim 1', labelpad=-10)
    ax.set_zlabel('Singular Dim 2', labelpad=-10)

    ax.set_xlim(left=np.min(np.asarray(transformations)[:, :, 0]), right=np.max(np.asarray(transformations)[:, :, 0]))
    ax.set_ylim(bottom=np.min(np.asarray(transformations)[:, :, 1]), top=np.max(np.asarray(transformations)[:, :, 1]))
    ax.set_zlim(bottom=np.min(np.asarray(transformations)[:, :, 2]), top=np.max(np.asarray(transformations)[:, :, 2]))

    angles = cycle(range(360))
    start_angle = 0

    # plot
    for tick in range(1, len(transformations)):

        # slowly rotate
        angle = next(angles)
        ax.view_init(30, angle + start_angle)

        # title
        title_pos = 1.0
        ax.set_title(f'{label}\nstep={tick * steps_in_tick}')
        ax.title.set_y(title_pos)  # otherwise the Axes3D object will lower it over time

        # lines
        for cat_id in range(num_cats):
            color = palette[cat_id]
            x, y, z = zip(*[transformation[cat_id] for transformation in transformations[:tick]])
            ax.plot(x, y, z, c=color, lw=config.Fig.line_width)  # x, y, z each have only 1 element

        # plot average location of all but last category
        if plot_avg_location_first_three_cats:
            xyz = np.expand_dims(transformations[tick][:-1].mean(0), axis=1)
            ax.scatter3D(*xyz, c=[palette[-1]], s=10)

        # visually mark that delay tick occurs
        if tick >= delay_tick:  # once shown, it stays
            ax.set_title(f'Category {num_cats}', loc='right')
            ax.title.set_y(1.0)  # otherwise the Axes3D object will lower it over time

        # save each fig individually, because celluloid.camera cannot deal with rotating axis
        file_path = images_path / f'{tick:0>6}.png'
        print(f'Saving {file_path}')
        plt.savefig(file_path)


def make_predictions_animation(outputs: np.ndarray,
                               label: str,
                               steps_in_tick: int,
                               delay_tick: int,
                               num_fragments: int,
                               images_path: Path,
                               ) -> None:
    """
    Saves figures, showing time course of predictions
    """

    assert np.ndim(outputs) == 3  # (ticks, words/categories, output dimension)

    num_ticks = outputs.shape[0]
    num_cats = outputs.shape[1]
    num_types = outputs.shape[2]
    palette = np.array(sns.color_palette("hls", num_cats))

    # fig
    fig, axarr = plt.subplots(num_cats, dpi=163)  # keep fig size small so that gif will animate
    cat_id2lines = {}
    x = np.arange(num_types)
    cat_id2text = {cat_id: None for cat_id in range(num_fragments)}
    text_bbox = dict(boxstyle="round", ec=(1., 0.5, 0.5), fc=(1., 0.8, 0.8), )

    # make a vertical line to visually separate v, w, x, y
    for cat_id in range(num_cats):
        for boundary in range(0, num_types, num_types // num_fragments):
            axarr[cat_id].axvline(x=boundary, linestyle=':', c='grey', lw=1)

    # plot
    for tick in range(num_ticks):

        plt.suptitle(f'{label}\nstep={tick * steps_in_tick}', y=1.0)

        for cat_id in range(num_cats):

            # axis
            axarr[cat_id].spines['right'].set_visible(False)
            axarr[cat_id].spines['top'].set_visible(False)
            axarr[cat_id].spines['bottom'].set_visible(False)
            axarr[cat_id].set_xticks([])
            axarr[cat_id].set_xticklabels([])
            axarr[cat_id].set_xlabel('')
            axarr[cat_id].set_ylabel('P(next word)', fontsize=8)
            axarr[cat_id].set_ylim(top=np.max(outputs))

            # remove previous points
            color = palette[cat_id]
            y = outputs[tick, cat_id]
            try:
                cat_id2lines[cat_id].remove()
            except KeyError:
                pass

            # draw points
            points = axarr[cat_id].scatter(x, y, c=[color], s=1)
            cat_id2lines[cat_id] = points

            # visually mark delay tick
            text = cat_id2text[cat_id]
            if tick < delay_tick:
                if text is not None:
                    text.remove()
                cat_id2text[cat_id] = axarr[cat_id].text(0, np.max(outputs) / 2, 'OFF', bbox=text_bbox)
            else:
                if text is not None:
                    text.remove()
                    cat_id2text[cat_id] = None

            # title
            axarr[cat_id].set_title(f'Category {cat_id + 1}', fontsize=8, loc='center')
            axarr[cat_id].title.set_y(0.75)

        # save each fig individually, because celluloid.camera cannot deal with rotating axis
        file_path = images_path / f'{tick:0>6}.png'
        print(f'Saving {file_path}')
        plt.savefig(file_path)


def plot_singular_values(ys: List[np.ndarray],
                         max_s: int,
                         label_name: str,
                         label_values: List[Any],
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
    ax.grid(True, axis='x')
    x = np.arange(max_s) + 1  # num columns
    if label_all_x:
        ax.set_xticks(x)
        ax.set_xticklabels(x)

    # plot
    colors = iter(['C0', 'C1', 'C3', 'C2', 'C4', 'C5', 'C6'])
    for n, y in enumerate(ys):
        color = next(colors)
        ax.plot(x,
                y,
                label=f'{label_name}={label_values[n]}',
                linewidth=2,
                color=color)
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

    fig, ax = plt.subplots(figsize=(8, 4), dpi=163)
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
        ax.plot(x, y, label=label, color=color, lw=2)
        # ax.scatter(x, y, color=color)
    if vline is not None:
        ax.axvline(x=vline, linestyle=':', color='grey', zorder=1)

    if legend:
        plt.legend(bbox_to_anchor=(1.0, 1.0), borderaxespad=1.0,
                   fontsize=config.Fig.leg_fs, frameon=False, loc='upper left', ncol=1)

    return fig