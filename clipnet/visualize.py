# visualize.py
# Adam He <adamyhe@gmail.com>

"""
Functions for plotting figures used in the CLIPNET paper. Many of these have been
adapted from elsewhere.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_side(arr, ylim=[-2, 2.5], yticks=[0, 2], xticks=[], pic_name=None):
    """
    Adapted from APARENT code (Bogard et al. 2019)
    """
    if arr.shape[0] % 2 != 0:
        raise ValueError("arr must have even length.")
    midpoint = int(arr.shape[0] / 2)
    pl = arr[:midpoint]
    mn = arr[midpoint:]
    plt.bar(
        range(pl.shape[0]),
        pl,
        width=-2,
        color="r",
    )
    plt.bar(range(mn.shape[0]), -mn, width=-2, color="b")
    axes = plt.gca()
    axes.set_ylim(ylim)
    axes.set_yticks(yticks)
    axes.set_xticks(xticks)
    axes.spines[["right", "top", "bottom"]].set_visible(False)
    plt.xlim(-0.5, pl.shape[0] - 0.5)
    axes.tick_params(labelleft=False)

    if pic_name is None:
        plt.show()
    else:
        plt.savefig(pic_name, transparent=True)
        plt.close()


def plot_side_stacked(
    arr0, arr1, ylim=[-1, 1], yticks=[0, 1], xticks=[], pic_name=None
):
    if arr0.shape[0] % 2 != 0 or arr1.shape[0] % 2 != 0:
        raise ValueError("arr must have even length.")
    midpoint = int(arr0.shape[0] / 2)
    pl0 = arr0[:midpoint]
    mn0 = arr0[midpoint:]
    pl1 = arr1[:midpoint]
    mn1 = arr1[midpoint:]
    plt.bar(range(pl0.shape[0]), pl0, width=2, color="tomato", alpha=0.5)
    plt.bar(range(mn0.shape[0]), -mn0, width=2, color="tomato", alpha=0.5)
    plt.bar(range(pl1.shape[0]), pl1, width=2, color="grey", alpha=0.5)
    plt.bar(range(mn1.shape[0]), -mn1, width=2, color="grey", alpha=0.5)
    axes = plt.gca()
    axes.set_ylim(ylim)
    axes.set_yticks(yticks)
    axes.set_xticks(xticks)
    axes.spines[["right", "top", "bottom"]].set_visible(False)
    plt.xlim(-0.5, pl0.shape[0] - 0.5)

    if pic_name is None:
        plt.show()
    else:
        plt.savefig(pic_name, transparent=True)
        plt.close()


def plot_a(ax, base, left_edge, height, color):
    """
    Adapted from DeepLIFT visualization code (Shrikumar et al. 2017)
    """
    a_polygon_coords = [
        np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.8],
                [0.2, 0.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.8],
                [0.8, 0.0],
            ]
        ),
        np.array(
            [
                [0.225, 0.45],
                [0.775, 0.45],
                [0.85, 0.3],
                [0.15, 0.3],
            ]
        ),
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(
            matplotlib.patches.Polygon(
                (
                    np.array([1, height])[None, :] * polygon_coords
                    + np.array([left_edge, base])[None, :]
                ),
                facecolor=color,
                edgecolor=color,
            )
        )


def plot_c(ax, base, left_edge, height, color):
    """
    Adapted from DeepLIFT visualization code (Shrikumar et al. 2017)
    """
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65, base + 0.5 * height],
            width=1.3,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65, base + 0.5 * height],
            width=0.7 * 1.3,
            height=0.7 * height,
            facecolor="white",
            edgecolor="white",
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 1, base],
            width=1.0,
            height=height,
            facecolor="white",
            edgecolor="white",
            fill=True,
        )
    )


def plot_g(ax, base, left_edge, height, color):
    """
    Adapted from DeepLIFT visualization code (Shrikumar et al. 2017)
    """
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65, base + 0.5 * height],
            width=1.3,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65, base + 0.5 * height],
            width=0.7 * 1.3,
            height=0.7 * height,
            facecolor="white",
            edgecolor="white",
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 1, base],
            width=1.0,
            height=height,
            facecolor="white",
            edgecolor="white",
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.825, base + 0.085 * height],
            width=0.174,
            height=0.415 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.625, base + 0.35 * height],
            width=0.374,
            height=0.15 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )


def plot_t(ax, base, left_edge, height, color):
    """
    Adapted from DeepLIFT visualization code (Shrikumar et al. 2017)
    """
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.4, base],
            width=0.2,
            height=height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge, base + 0.8 * height],
            width=1.0,
            height=0.2 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )


default_colors = {0: "green", 1: "blue", 2: "orange", 3: "red"}
default_plot_funcs = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t}


def plot_weights_given_ax(
    ax,
    array,
    pos_height,
    neg_height,
    length_padding,
    subticks_frequency,
    highlight,
    colors=default_colors,
    plot_funcs=default_plot_funcs,
):
    """
    Adapted from DeepLIFT visualization code (Shrikumar et al. 2017)
    """
    if len(array.shape) == 3:
        array = np.squeeze(array)
    if array.shape[0] % 2 != 0:
        raise ValueError("arr must have even length.")
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)
    if array.shape[1] % 4 != 0:
        raise ValueError("Incorrect number of nucleotide dummy variables.")
    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    for i in range(array.shape[0]):
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if letter[1] > 0:
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            plot_func(
                ax=ax, base=height_so_far, left_edge=i, height=letter[1], color=color
            )
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            if start_pos < 0.0 or end_pos > array.shape[0]:
                raise ValueError(
                    "Highlight positions must be within the bounds of the sequence."
                )
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    xy=[start_pos, min_depth],
                    width=end_pos - start_pos,
                    height=max_height - min_depth,
                    edgecolor=color,
                    fill=False,
                )
            )

    ax.set_xlim(-length_padding, array.shape[0] + length_padding)
    ax.xaxis.set_ticks(np.arange(0.0, array.shape[0] + 1, subticks_frequency))
    ax.set_ylim(neg_height, pos_height)


def plot_weights(
    array,
    figsize=(20, 2),
    pos_height=1.0,
    neg_height=-1.0,
    length_padding=1.0,
    subticks_frequency=1.0,
    colors=default_colors,
    plot_funcs=default_plot_funcs,
    highlight={},
):
    """
    Adapted from DeepLIFT visualization code (Shrikumar et al. 2017)
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_weights_given_ax(
        ax=ax,
        array=array,
        pos_height=pos_height,
        neg_height=neg_height,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight,
    )
