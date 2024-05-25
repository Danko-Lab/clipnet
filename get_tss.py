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
import clipnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fasta_fp",
        type=str,
        default=None,
        help="If pyfastx throws an error, try deleting .fxi index files.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="where should the output be written? Will export a joblib.gz file.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ensemble_models/",
        help="directory where to load models from.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Index of GPU to use (starting at 0). If None, will use CPU.",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        nn = clipnet.CLIPNET(n_gpus=1, use_specific_gpu=args.gpu)
    else:
        nn = clipnet.CLIPNET(n_gpus=0)
    tss = nn.compute_tss(args.model_dir, args.fasta_fp)
    joblib.dump(tss, args.output)


if __name__ == "__main__":
    main()
