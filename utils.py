"""
Important helper functions for CLIPNET (mostly data loading and plotting).
"""

import gzip
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyfastx
import tqdm


class TwoHotDNA:
    """
    Allows you to access id, seq, and twohot(seq) as attributes. Handles IUPAC ambiguity
    codes for heterozygotes.
    """

    def __init__(self, record):
        # add attributes to self
        if hasattr(record, "id") and hasattr(record, "seq"):
            self.id = record.id
            self.seq = record.seq
        else:
            self.seq = record
        # get sequence into an array
        seq_list = list(self.seq.upper())
        # one hot the sequence
        encoding = {
            "A": np.array([2, 0, 0, 0]),
            "C": np.array([0, 2, 0, 0]),
            "G": np.array([0, 0, 2, 0]),
            "T": np.array([0, 0, 0, 2]),
            "N": np.array([0, 0, 0, 0]),
            "M": np.array([1, 1, 0, 0]),
            "R": np.array([1, 0, 1, 0]),
            "W": np.array([1, 0, 0, 1]),
            "S": np.array([0, 1, 1, 0]),
            "Y": np.array([0, 1, 0, 1]),
            "K": np.array([0, 0, 1, 1]),
        }
        twohot = [encoding.get(seq, seq) for seq in seq_list]
        self.twohot = np.array(twohot)


class RevTwoHotDNA:
    """
    Reverses an twohot encoding into a string. Handles IUPAC ambiguity codes for
    heterozygotes. Assumes array is (bp, 4).
    """

    def __init__(self, twohot, name=None):
        # add attributes to self
        self.twohot = twohot
        self.name = name
        self.id = name
        # reverse one hot the sequence
        encoding = {
            "A": np.array([2, 0, 0, 0]),
            "C": np.array([0, 2, 0, 0]),
            "G": np.array([0, 0, 2, 0]),
            "T": np.array([0, 0, 0, 2]),
            "N": np.array([0, 0, 0, 0]),
            "M": np.array([1, 1, 0, 0]),
            "R": np.array([1, 0, 1, 0]),
            "W": np.array([1, 0, 0, 1]),
            "S": np.array([0, 1, 1, 0]),
            "Y": np.array([0, 1, 0, 1]),
            "K": np.array([0, 0, 1, 1]),
        }
        reverse_encoding = {encoding[k].tobytes(): k for k in encoding.keys()}
        seq = [reverse_encoding[np.array(pos).tobytes()] for pos in twohot.tolist()]
        self.seq = "".join(seq)


def get_twohot(seq):
    """Extracts just the twohot encoding from TwoHotDNA."""
    return TwoHotDNA(seq).twohot


def get_twohot_from_series(series, cores=8, desc="Twohot encoding", silence=False):
    """Extracts just the twohot encoding from TwoHotDNA.

    Given a pandas series of sequences, returns an twohot-encoded array (n, len, 4)
    array of all sequences.
    """
    if cores > 1:
        # Use multiprocessing to parallelize twohot encoding
        import multiprocessing as mp

        pool = mp.Pool(min(cores, mp.cpu_count()))
        twohot_encoded = list(
            tqdm.tqdm(
                pool.imap(get_twohot, series.to_list()),
                total=len(series),
                desc=desc,
                disable=silence,
            )
        )
    else:
        twohot_encoded = [
            get_twohot(seq)
            for seq in tqdm.tqdm(series.to_list(), desc=desc, disable=silence)
        ]
    return np.array(twohot_encoded)


def gz_read(fp):
    """Handles opening gzipped or non-gzipped files to read mode."""
    ext = os.path.splitext(fp)[-1]
    if ext == ".gz" or ext == ".bgz":
        return gzip.open(fp, mode="r")
    else:
        return open(fp, mode="r")


def all_equal(x):
    """
    Returns whether all entries in an iterable are the same (number of entries equal
    to first is len).
    """
    return x.count(x[0]) == len(x)


def numerical_sort(value):
    """Sort a list of strings numerically."""
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def list_split(listylist, n):
    """Split a list l into smaller lists of size n."""
    # For item i in a range that is a length of l,
    for i in range(0, len(listylist), n):
        # Create an index range for l of n items:
        yield listylist[i : i + n]


def split_window_indices_by_experiment(list_of_indices):
    """
    Reconfigures a list of indices [(experiment, window_index_in_bed)] to a
    dictionary, where the keys are the experiments and the values are lists of each all
    the windows present in that experiment in the original list.
    """
    split_list = {}
    for index in list_of_indices:
        if index[0] in split_list.keys():
            split_list[index[0]].append(index[1])
        else:
            split_list[index[0]] = [index[1]]
    return split_list


def get_bedtool_from_list(bt, list_of_ints):
    return [bt[i] for i in list_of_ints]


def get_twohot_fasta_sequences(
    fasta_fp, cores=8, desc="Twohot encoding", silence=False
):
    """
    Given a fasta file with each record, returns a twohot-encoded array (n, len, 4)
    array of all sequences.
    """
    fa = pyfastx.Fasta(fasta_fp)
    seqs = [
        rec.seq
        for rec in tqdm.tqdm(
            fa, desc="Reading sequences", disable=silence, total=len(fa)
        )
    ]
    if cores > 1:
        # Use multiprocessing to parallelize twohot encoding
        import multiprocessing as mp

        pool = mp.Pool(min(cores, mp.cpu_count()))
        twohot_encoded = list(
            tqdm.tqdm(
                pool.imap(get_twohot, seqs), total=len(seqs), desc=desc, disable=silence
            )
        )
    else:
        twohot_encoded = [
            TwoHotDNA(seq).twohot for seq in tqdm.tqdm(seqs, desc=desc, disable=silence)
        ]
    return np.array(twohot_encoded)


def get_consensus_region(bed_intervals, consensus_fp):
    """
    Given a list of bed intervals and a consensus.fna file path, get list of sequences
    as strings.
    """
    sequences = []
    fna = pyfastx.Fasta(consensus_fp)
    for interval in bed_intervals:
        # Recall that pyfastx uses 1 based [) half open encoding.
        sequences.append(fna.fetch(interval.chrom, (interval.start + 1, interval.stop)))
    return sequences


def get_consensus_twohot(bed_intervals, consensus_fp):
    """
    Given a list of bed intervals and a consensus.fna file path, return a list of
    twohot encodings.
    """
    sequences = get_consensus_region(bed_intervals, consensus_fp)
    twohot_list = [TwoHotDNA(sequence).twohot for sequence in sequences]
    return twohot_list


def rc_twohot_het(arr):
    """
    Computes reverse-complement twohot. Handles heterozygotes encoded via IUPAC
    ambiguity codes.
    """
    # inverting each sequence in arr_rc along both axes takes the reverse complement.
    # Except for the at and cg heterozygotes, which need to be complemented by masks.
    arr_rc = np.array([seq[::-1, ::-1] for seq in arr])
    # Get mask of all at and cg heterozygotes
    at = np.all(arr_rc == [1, 0, 0, 1], axis=2)
    cg = np.all(arr_rc == [0, 1, 1, 0], axis=2)
    # Complement at and cg heterozygotes
    arr_rc[at] = [0, 1, 1, 0]
    arr_rc[cg] = [1, 0, 0, 1]
    return arr_rc


def slice_procap(procap, pad):
    """
    Slices the procap_chunk to the middle with pad. Handles both single and double
    strand cases.
    """
    if procap.shape[0] == 0:
        return procap
    else:
        dim = procap.shape[1]
        slc = np.r_[pad : int(dim / 2) - pad, int(dim / 2) + pad : dim - pad]
        return procap[:, slc]


def check_dimensions(seq, procap):
    """Check that dimensions are correct."""
    if seq.shape[0] != procap.shape[0]:
        raise ValueError(f"n_samples: seq={seq.shape[0]}, procap={procap.shape[0]}.")
    if seq.shape[1] != procap.shape[1] / 2:
        raise ValueError(f"len(windows): seq={seq.shape[1]}, procap={procap.shape[1]}.")
    if seq.shape[2] != 4:
        raise ValueError(f"seq dummy variables = {seq.shape[2]}.")


# The following functions are adapted from DeepLIFT and https://alextseng.net/blog/posts/20201122-kmer-shuffles/:


def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytes(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    if arr.dtype != np.int8:
        raise ValueError("Array must be of type np.int8")
    return arr.tostring().decode("ascii")


def one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens


def tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]


def kshuffle(seq, num_shufs=1, k=2, random_seed=None):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    Arguments:
        `seq`: either a string of length L.
        `num_shufs`: the number of shuffles to create, N
        `k`: the length k-mer whose frequencies are to be preserved; defaults
            to k = 2 (i.e. preserve dinucleotide frequencies)
        `rng`: a NumPy RandomState object, to use for performing shuffles
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`.
    """
    # Convert the sequence (string) into a 1D array of numbers (for simplicity)
    if isinstance(seq, str):
        arr = string_to_char_array(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    rng = np.random.RandomState(random_seed)

    if k == 1:
        # Do simple shuffles of `arr`
        all_results = []
        for i in range(num_shufs):
            rng.shuffle(arr)
            all_results.append(char_array_to_string(arr))
        return all_results

    # Tile `arr` from a 1D array to a 2D array of all (k-1)-mers (i.e.
    # "shortmers"), using -1 as a "sentinel" for the last few values
    arr_shortmers = np.empty((len(arr), k - 1), dtype=arr.dtype)
    arr_shortmers[:] = -1
    for i in range(k - 1):
        arr_shortmers[: len(arr) - i, i] = arr[i:]

    # Get the set of all shortmers, and a mapping of which positions start with
    # which shortmers; `tokens` is the mapping, and is an integer representation
    # of the original shortmers (range [0, # unique shortmers - 1])
    shortmers, tokens = np.unique(arr_shortmers, return_inverse=True, axis=0)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for token in range(len(shortmers)):
        # Locations in `arr` where the shortmer exists; some shortmers will have
        # the sentinel, but that's okay
        mask = tokens == token
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 to indices for next token

    all_results = []

    for i in range(num_shufs):
        # Shuffle the next indices
        for t in range(len(shortmers)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(shortmers)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        shuffled_arr = shortmers[result][:, 0]  # First character of each shortmer
        # (this leaves behind the sentinels)

        all_results.append(char_array_to_string(shuffled_arr))
    return all_results


def save_dict_to_hdf5(file, group, data, compression="gzip"):
    for key, value in data.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            save_dict_to_hdf5(file, subgroup, value)
        else:
            file.create_dataset(
                f"{group.name}/{key}", data=value, compression=compression
            )


def l2_score(x, y):
    return np.sqrt(np.sum(np.square(x - y), axis=1))


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
    if arr.shape[0] % 2 != 0:
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
