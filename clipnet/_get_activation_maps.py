#!/usr/bin/env python3

"""
This script calculates TSS position weight matrices from a fit clipnet.py model.
"""

import argparse
import logging
import os

import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
from . import clipnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_fp", type=str, help="file path to model fold to load.")
    parser.add_argument(
        "fasta_fp",
        type=str,
        default=None,
        help="If pyfastx throws an error, try deleting .fxi index files.",
    )
    parser.add_argument(
        "predicted_tss_fp",
        type=str,
        help="Where to load predicted TSS positions from.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="where should the output be written? Will export a joblib.gz file.",
    )
    parser.add_argument(
        "--conv_layer",
        type=int,
        default=1,
        help="Which conv layer to get activations for",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=200,
        help="how wide of a window around tss to select.",
    )
    args = parser.parse_args()

    nn = clipnet.CLIPNET(n_gpus=0)
    activations = nn.get_activation_maps(
        args.model_fp,
        args.fasta_fp,
        args.predicted_tss_fp,
        layer=args.conv_layer,
        window=args.window,
    )
    joblib.dump(activations, args.output)


if __name__ == "__main__":
    main()
