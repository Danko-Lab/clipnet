#!/usr/bin/env python3

"""
This script calculates performance metrics given a set of predictions and ground truth.
"""

import argparse

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "predictions",
        type=str,
        help="An hdf5 file containing predictions (track, quantity).",
    )
    parser.add_argument(
        "observed",
        type=str,
        help="A csv(.gz), npy, or npz file containing the observed procap tracks.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="An hdf5 file to write the performance metrics to.",
    )
    args = parser.parse_args()
    with h5py.File(args.predictions, "r") as hf:
        track, quantity = hf["track"][:], hf["quantity"][:]

    if args.observed.endswith(".npz") or args.observed.endswith(".npy"):
        observed = np.load(args.observed)
        if args.observed.endswith(".npz"):
            observed = observed["arr_0"]
    elif args.observed.endswith(".csv.gz") or args.observed.endswith(".csv"):
        observed = pd.read_csv(args.observed).to_numpy()
    assert (
        track.shape[0] == observed.shape[0]
    ), f"n predictions ({track.shape[0]}) and n observed ({observed.shape[0]}) do not match."
    assert (
        track.shape[1] <= observed.shape[1]
    ), f"Predicted tracks ({track.shape[1]}) are longer than observed ({observed.shape[1]})."
    assert (
        observed.shape[1] - track.shape[1]
    ) % 4 == 0, "Padding around predicted tracks must be divisible by 4."
    start = (observed.shape[1] - track.shape[1]) // 4
    end = observed.shape[1] // 2 - start
    observed_clipped = observed[
        :,
        np.r_[start:end, observed.shape[1] // 2 + start : observed.shape[1] // 2 + end],
    ]

    track_pearson = pd.DataFrame(track[0]).corrwith(
        pd.DataFrame(observed_clipped), axis=1
    )
    track_js_distance = jensenshannon(track[0], observed_clipped, axis=1)
    quantity_log_pearson = pearsonr(np.log1p(quantity), np.log1p(observed.sum(axis=1)))
    quantity_spearman = spearmanr(quantity, observed.sum(axis=1))

    print(f"Median Track Pearson: {track_pearson.median():.4f}")
    print(f"Median Track JS Distance: {track_js_distance.median():.4f}")
    print(f"Quantity Log Pearson: {quantity_log_pearson[0]:.4f}")
    print(f"Quantity Spearman: {quantity_spearman[0]:.4f}")

    with h5py.File(args.output, "w") as hf:
        hf.create_dataset(
            "track_pearson", data=track_pearson.to_numpy(), compression="gzip"
        )
        hf.create_dataset(
            "track_js_distance", data=track_js_distance, compression="gzip"
        )
        hf.create_dataset(
            "quantity_log_pearson", data=quantity_log_pearson[0], compression="gzip"
        )
        hf.create_dataset(
            "quantity_spearman", data=quantity_spearman[0], compression="gzip"
        )


if __name__ == "__main__":
    main()
