"""
This file contains a number of functions and the class CGen (CLIPNET Generator) that assist in loading data for the
CLIPNET project.
"""

import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import utils


def load_track(fp, unpackbits=False):
    if os.path.splitext(fp)[-1] == ".npz":
        arr = np.load(fp)["arr_0"]
        if unpackbits:
            return np.unpackbits(arr, axis=1)
        else:
            return arr
    else:
        return np.array(pd.read_csv(fp, index_col=0, header=None))


def load_data(seq_fp, dnase_fp, procap_fp, pad=0, reverse_complement=False):
    """
    Load a single, unchunked dataset. Use reverse_complement=True to load dataset
    reverse complemented.
    """
    # check if string.
    if (
        isinstance(seq_fp, str)
        and isinstance(dnase_fp, str)
        and isinstance(procap_fp, str)
    ):
        pass
    # otherwise, check if len 1 iterables and use the first item.
    else:
        assert (
            len(seq_fp) == len(dnase_fp) == len(procap_fp) == 1
        ), "seq_fp, dnase_fp, procap_fp must be strings or singleton iterables."
        try:
            seq_fp = seq_fp[0]
            dnase_fp = dnase_fp[0]
            procap_fp = procap_fp[0]
        except TypeError:
            print("seq_fp, dnase_fp, procap_fp must be strings or singleton iterables.")
    # load data and check dimensions
    print("Loading data from %s, %s, and %s ..." % (seq_fp, dnase_fp, procap_fp))
    seq = utils.get_onehot_fasta_sequences(seq_fp)
    dnase = load_track(dnase_fp, unpackbits=True)
    procap = load_track(procap_fp)
    utils.check_dimensions(seq, dnase, procap)
    print("Successfully loaded data")
    # do rc_augmentation
    if reverse_complement:
        seq = utils.rc_onehot_het(seq)
        dnase = dnase[:, ::-1]
        procap = procap[:, ::-1]
        print("Computed reverse complement.")
    # output datasets
    X = np.dstack((seq, dnase))
    y = utils.slice_procap(procap, pad)
    return X, y


class CGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        seq_chunks,
        dnase_chunks,
        procap_chunks,
        steps_per_epoch,
        batch_size,
        pad,
        rc_augmentation=True,
        intershuffle=True,
        intrashuffle=True,
    ):
        # Check that lists of chunks are all the same
        assert (
            len(seq_chunks) == len(dnase_chunks) == len(procap_chunks)
        ), "lengths: seq_chunks = %d, dnase_chunks = %d, procap_chunks = %d." % (
            len(seq_chunks),
            len(dnase_chunks),
            len(procap_chunks),
        )
        self.seq_chunks = seq_chunks
        self.dnase_chunks = dnase_chunks
        self.procap_chunks = procap_chunks
        self.chunk_list = np.arange(len(self.seq_chunks))
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.pad = pad
        self.rc_augmentation = rc_augmentation
        self.intershuffle = intershuffle
        self.intrashuffle = intrashuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.steps_per_epoch

    def on_epoch_end(self):
        """Shuffles chunk indexes on start and after each epoch."""
        if self.intershuffle and len(self.chunk_list) > 1:
            random.shuffle(self.chunk_list)
        self.chunki = 0
        self.stepi = 0

    def __load_chunk(self):
        """Loads data from chunk at chunki index"""
        assert hasattr(self, "chunki"), "chunki is undefined when trying to load chunk."
        # Get filepaths from list
        rand_chunki = self.chunk_list[self.chunki]
        seq_chunk_fp = self.seq_chunks[rand_chunki]
        dnase_chunk_fp = self.dnase_chunks[rand_chunki]
        procap_chunk_fp = self.procap_chunks[rand_chunki]

        # Load chunk data from disk
        seq_chunk = utils.get_onehot_fasta_sequences(seq_chunk_fp)
        dnase_chunk = np.unpackbits(np.load(dnase_chunk_fp)["arr_0"], axis=1)
        procap_chunk = np.load(procap_chunk_fp)["arr_0"]
        utils.check_dimensions(seq_chunk, dnase_chunk, procap_chunk)

        # Perform reverse complement augmentation
        if self.rc_augmentation:
            seq_chunk = np.concatenate([seq_chunk, utils.rc_onehot_het(seq_chunk)])
            dnase_chunk = np.concatenate([dnase_chunk, dnase_chunk[:, ::-1]])
            procap_chunk = np.concatenate([procap_chunk, procap_chunk[:, ::-1]])

        # Split list of ids into batches
        chunk_ids = np.arange(seq_chunk.shape[0])
        if self.intrashuffle:
            random.shuffle(chunk_ids)
        self.batches = list(utils.list_split(chunk_ids, self.batch_size))

        # Save chunk data to self.
        self.X_chunk = np.dstack((seq_chunk, dnase_chunk))
        sliced_procap_chunk = utils.slice_procap(procap_chunk, self.pad)
        self.y_chunk = sliced_procap_chunk

        # Reset batchi
        self.batchi = 0
        # Increment chunk counter (chunki gets reset by on_epoch_end)
        self.chunki += 1

    def __getitem__(self, index):
        """Gets a batch of data. Uses custom counters stepi and chunki instead of index."""
        # stepi == 0 means new epoch, batchi == n_batches means end of chunk.
        if self.stepi == 0 or self.batchi == len(self.batches):
            self.__load_chunk()

        # Get the batch ids
        batch = self.batches[self.batchi]
        # Increment batch counter (batchi gets reset by __load_chunk)
        self.batchi += 1
        # Increment step counter (stepi gets reset by on_epoch_end)
        self.stepi += 1

        # Extract batch from chunk and return
        X = self.X_chunk[batch, :, :]
        y_batch = self.y_chunk[batch, :]
        y_sum = y_batch.sum(axis=1)
        # y_profile = y_batch / y_sum
        y = [y_batch, y_sum]  # for old behavior just use [y_batch, y_sum]
        return X, y
