## NOT IMPLEMENTED

"""
This script calculates TSS position weight matrices from a fit clipnet.py model.
"""

import argparse
import logging
import os

import pandas as pd

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
        "output",
        type=str,
        help="where should the output be written? Will export a csv(.gz) file.",
    )
    parser.add_argument(
        "--conv_layer",
        type=int,
        default=1,
        help="Which conv layer to get activations for",
    )
    parser.add_argument(
        "--filter_width",
        type=int,
        default=15,
        help="how wide is the width of each filter in this layer?",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5000,
        help="what is the number of top activating subsequences should we use to calculate GC content?",
    )
    args = parser.parse_args()

    nn = clipnet.CLIPNET(n_gpus=0)
    gc_content = nn.get_filter_gc_content(
        args.model_fp,
        args.fasta_fp,
        layer=args.conv_layer,
        filter_width=args.filter_width,
        n=args.n,
    )
    pd.Series(gc_content).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
