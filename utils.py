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


def twohot_fasta(fasta_fp, cores=16):
    """
    Given a fasta file with each record, returns an onehot-encoded array (n, len, 4)
    array of all sequences.

    ### NOTE: THIS IS A RENAME OF get_onehot_fasta_sequences(). I originally called
    ### our encoding onehot, but it's too late to start renaming everything now. :(
    ### This function is used to make the API less of a mess.
    """
    return get_onehot_fasta_sequences(fasta_fp, cores=cores)


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


# The following functions are adapted from DeepLIFT and https://alextseng.net/blog/posts/20201122-kmer-shuffles/:

def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytes(seq, "utf8"), dtype=np.int8)
        precis_buckets = np.zeros_like(precis)
        np.put_along_axis(precis_buckets, thresh_buckets, precis, -1)

def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    assert arr.dtype == np.int8
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
    if type(seq) is str:
        arr = string_to_char_array(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")

    rng = np.random.RandomState(random_seed)

    if k == 1:
        # Do simple shuffles of `arr`
        all_results = []
        for i in range(num_shufs):
            rng.shuffle(arr)
            if type(seq) is str:
                all_results.append(char_array_to_string(arr))
            else:
                all_results[i] = tokens_to_one_hot(arr, one_hot_dim)
        return all_results

    # Tile `arr` from a 1D array to a 2D array of all (k-1)-mers (i.e.
    # "shortmers"), using -1 as a "sentinel" for the last few values
    arr_shortmers = np.empty((len(arr), k - 1), dtype=arr.dtype)
    arr_shortmers[:] = -1
    for i in range(k - 1):
        arr_shortmers[:len(arr) - i, i] = arr[i:]

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