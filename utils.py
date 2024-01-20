"""
Important helper functions for clipnet_generator.
"""

import gzip
import os
import re

import numpy as np
import pyfastx


class OneHotDNA:
    """
    Allows you to access id, seq, and onehot(seq) as attributes. Handles IUPAC ambiguity
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
        onehot = [encoding.get(seq, seq) for seq in seq_list]
        self.onehot = np.array(onehot)


class RevOneHotDNA:
    """
    Reverses an onehot encoding into a string. Handles IUPAC ambiguity codes for
    heterozygotes. Assumes array is (bp, 4).
    """

    def __init__(self, onehot, name=None):
        # add attributes to self
        self.onehot = onehot
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

        seq = [reverse_encoding[np.array(pos).tobytes()] for pos in onehot.tolist()]
        self.seq = "".join(seq)


def get_onehot(seq):
    """Extracts just the onehot encoding from OneHotDNA."""
    return OneHotDNA(seq).onehot


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


def get_onehot_fasta_sequences(fasta_fp, cores=16):
    """
    Given a fasta file with each record, returns an onehot-encoded array (n, len, 4)
    array of all sequences.
    """
    seqs = [rec.seq for rec in pyfastx.Fasta(fasta_fp)]
    if cores > 1:
        # Use multiprocessing to parallelize onehot encoding
        import multiprocessing as mp

        pool = mp.Pool(min(cores, mp.cpu_count()))
        parallelized = pool.map(get_onehot, seqs)
        onehot_encoded = np.array([p for p in parallelized])
    else:
        onehot_encoded = np.array([OneHotDNA(seq).onehot for seq in seqs])
    return onehot_encoded


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


def get_consensus_onehot(bed_intervals, consensus_fp):
    """
    Given a list of bed intervals and a consensus.fna file path, return a list of
    onehot encodings.
    """
    sequences = get_consensus_region(bed_intervals, consensus_fp)
    onehot_list = [OneHotDNA(sequence).onehot for sequence in sequences]
    return onehot_list


def rc_onehot_het(arr):
    """
    Computes reverse-complement onehot. Handles heterozygotes encoded via IUPAC
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


def check_dimensions(seq, procap, dnase=None):
    """Check that dimensions are correct. DNase will be ignored if it is None."""
    assert (
        seq.shape[0] == procap.shape[0]
    ), f"n_samples: seq={seq.shape[0]}, procap={procap.shape[0]}."
    assert (
        seq.shape[1] == procap.shape[1] / 2
    ), f"len(windows): seq={seq.shape[1]}, procap={procap.shape[1]}."
    if dnase is not None:
        assert (
            seq.shape[0] == dnase.shape[0]
        ), f"n_samples: seq,procap={seq.shape[0]}, dnase={dnase.shape[0]}"
        assert (
            seq.shape[1] == dnase.shape[1] == procap.shape[1] / 2
        ), f"len(windows): seq,procap={seq.shape[1]}, dnase={dnase.shape[1]}"
    assert seq.shape[2] == 4, "seq dummy variables = %d." % seq.shape[2]
