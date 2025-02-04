# utils.py
# Adam He <adamyhe@gmail.com>

"""
Important helper functions for CLIPNET (mostly data loading and plotting).
"""

import gzip
import os
import re

import numpy as np
import pandas as pd
import pyfastx
import tqdm


def get_mut_bases(base):
    return [mut for mut in ["A", "C", "G", "T"] if mut != base]


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


def get_twohot_from_series(iterable, cores=8, silence=False):
    """Extracts just the twohot encoding from TwoHotDNA.

    Given a pandas series of sequences, returns an twohot-encoded array (n, len, 4)
    array of all sequences.
    """
    import multiprocessing as mp

    cores = min(cores, mp.cpu_count())
    if cores > 1:
        # Use multiprocessing to parallelize twohot encoding
        pool = mp.Pool(cores)
        twohot_encoded = list(
            tqdm.tqdm(
                pool.imap(get_twohot, iterable), desc="Twohot encoding", disable=silence
            )
        )
    else:
        twohot_encoded = [
            get_twohot(seq)
            for seq in tqdm.tqdm(iterable, desc="Twohot encoding", disable=silence)
        ]
    return np.array(twohot_encoded)


def extract_loci(
    fasta_fname,
    bed_fname,
    chroms=None,
    in_window=1000,
    cores=8,
    silence=False,
):
    """
    Fetches sequences from a fasta file given a bed file. Returns a twohot encoded
    array.

    Parameters
    ----------
    fasta_fname : str
        Path to fasta file
    bed_fname : str
        Path to bed file
    chroms : list, optional
        List of chromosomes to extract, by default None
    in_window : int, optional
        Size of window to extract, by default 1000
    cores : int, optional
        Number of cores to use, by default 8
    desc : str, optional
        Description for tqdm, by default "Extracting loci"
    silence : bool, optional
        Whether to silence tqdm, by default False
    """
    fa = pyfastx.Fasta(fasta_fname)
    loci = pd.read_csv(bed_fname, index_col=None, header=None, sep="\t")
    if chroms is not None:
        loci = loci[loci.iloc[:, 0].isin(chroms)]
    centers = (loci.iloc[:, 1] + loci.iloc[:, 2]) // 2
    windows = pd.DataFrame(
        {
            "chrom": loci.iloc[:, 0],
            "start": centers - in_window // 2,
            "end": centers + in_window // 2,
        }
    )
    seqs = [
        fa.fetch(windows.iloc[i, 0], (windows.iloc[i, 1], windows.iloc[i, 2]))
        for i in tqdm.tqdm(range(len(windows)), desc="Extracting loci", disable=silence)
    ]
    return get_twohot_from_series(seqs, cores=cores, silence=silence)


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
