"""
This file contains a number of functions and the class CGen (CLIPNET Generator) that
assist in loading data while training CLIPNET models.
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


def load_data(seq_fp, procap_fp, pad=0, reverse_complement=False):
    """
    Load a single, unfolded dataset. Use reverse_complement=True to load dataset
    reverse complemented.
    """
    # check if string.
    if isinstance(seq_fp, str) and isinstance(procap_fp, str):
        pass
    # otherwise, check if len 1 iterables and use the first item.
    else:
        if not len(seq_fp) == len(procap_fp) == 1:
            raise ValueError(
                "seq_fp and procap_fp must be strings or singleton iterables."
            )
        try:
            seq_fp = seq_fp[0]
            procap_fp = procap_fp[0]
        except TypeError:
            print("seq_fp, dnase_fp, procap_fp must be strings or singleton iterables.")
    # load data and check dimensions
    print(f"Loading sequence data from {seq_fp} and procap data from {procap_fp}")
    X = utils.get_twohot_fasta_sequences(seq_fp)
    procap = load_track(procap_fp)
    utils.check_dimensions(X, procap)
    print("Successfully loaded data")
    # do rc_augmentation
    if reverse_complement:
        X = utils.rc_twohot_het(X)
        procap = procap[:, ::-1]
        print("Computed reverse complement.")
    # output datasets
    y = utils.slice_procap(procap, pad)
    return X, y


class CGen(tf.keras.utils.Sequence):
    def __init__(
        self,
        seq_folds,
        procap_folds,
        steps_per_epoch,
        batch_size,
        pad,
        rc_augmentation=True,
        intershuffle=True,
        intrashuffle=True,
    ):
        # Check that lists of folds are all the same
        if len(seq_folds) != len(procap_folds):
            raise ValueError(
                f"lengths: seq_folds = {seq_folds}, procap_folds = {procap_folds}."
            )
        self.seq_folds = seq_folds
        self.procap_folds = procap_folds
        self.fold_list = np.arange(len(self.seq_folds))
        # print(self.seq_folds)
        # print(f"Loaded {len(self.fold_list)} folds.")
        # print(self.fold_list)
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.pad = pad
        self.rc_augmentation = rc_augmentation
        self.intershuffle = intershuffle
        self.intrashuffle = intrashuffle
        self.foldi = 0
        self.stepi = 0
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.steps_per_epoch

    def on_epoch_end(self):
        """Shuffles fold indexes on start and after each epoch."""
        if self.intershuffle and len(self.fold_list) > 1:
            random.shuffle(self.fold_list)
        self.foldi = 0
        self.stepi = 0

    def __load_fold(self):
        """Loads data from fold at foldi index"""
        assert hasattr(self, "foldi"), "foldi is undefined when trying to load fold."
        # Get filepaths from list
        # print(self.fold_list)
        rand_foldi = self.fold_list[self.foldi]
        seq_fold_fp = self.seq_folds[rand_foldi]
        procap_fold_fp = self.procap_folds[rand_foldi]

        # Load fold data from disk
        seq_fold = np.load(seq_fold_fp)["arr_0"]
        procap_fold = load_track(procap_fold_fp)
        utils.check_dimensions(seq_fold, procap_fold)

        # Perform reverse complement augmentation
        if self.rc_augmentation:
            seq_fold = np.concatenate([seq_fold, utils.rc_twohot_het(seq_fold)])
            procap_fold = np.concatenate([procap_fold, procap_fold[:, ::-1]])

        # Split list of ids into batches
        fold_ids = np.arange(seq_fold.shape[0])
        if self.intrashuffle:
            random.shuffle(fold_ids)
        self.batches = list(utils.list_split(fold_ids, self.batch_size))

        # Save fold data to self.
        self.X_fold = seq_fold
        sliced_procap_fold = utils.slice_procap(procap_fold, self.pad)
        self.y_fold = sliced_procap_fold

        # Reset batchi
        self.batchi = 0
        # Increment fold counter (foldi gets reset by on_epoch_end)
        self.foldi = min(len(self.fold_list) - 1, self.foldi + 1)

    def __getitem__(self, index):
        """
        Gets a batch of data. Uses custom counters stepi and foldi instead of index.
        Index arg is retained for compatibility with tf.keras.utils.Sequence.
        """
        # stepi == 0 means new epoch, batchi == n_batches means end of fold.
        if self.stepi == 0 or self.batchi == len(self.batches):
            self.__load_fold()

        # Get the batch ids
        batch = self.batches[self.batchi]
        # Increment batch counter (batchi gets reset by __load_fold)
        self.batchi += 1
        # Increment step counter (stepi gets reset by on_epoch_end)
        self.stepi += 1

        # Extract batch from fold and return
        X = self.X_fold[batch, :, :]
        y_batch = self.y_fold[batch, :]
        y_sum = y_batch.sum(axis=1)
        # y_profile = y_batch / y_sum
        y = [y_batch, y_sum]  # for old behavior just use [y_batch, y_sum]
        return X, y
