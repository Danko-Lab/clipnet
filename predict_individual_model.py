#!/usr/bin/env python3

"""
This script predicts on a set of data using a fitted model from clipnet.py.
"""

import argparse
import logging
import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import clipnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_fp", type=str, help="model file path.")
    parser.add_argument("fasta_fp", type=str, help="fasta file.")
    parser.add_argument("output_fp", type=str, help="output file path.")
    parser.add_argument(
        "--outputs",
        type=int,
        default=2,
        choices=[1, 2],
        help="Number of outputs for the model. Default is 2 for standard CLIPNET models "
        "and 1 for the single scalar output models (e.g. PauseNet, OrientNet).",
    )
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        help="reverse complements the input data.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Index of GPU to use (starting from 0). If not invoked, uses CPU.",
    )
    parser.add_argument(
        "--silence",
        action="store_true",
        help="disables progress bars and other non-essential print statements.",
    )
    args = parser.parse_args()

    nn = (
        clipnet.CLIPNET(n_gpus=1, use_specific_gpu=args.gpu)
        if args.gpu is not None
        else clipnet.CLIPNET(n_gpus=0)
    )
    prediction = nn.predict_on_fasta(
        model_fp=args.model_fp,
        fasta_fp=args.fasta_fp,
        outputs=args.outputs,
        reverse_complement=args.reverse_complement,
        low_mem=True,
        silence=args.silence,
    )
    if args.outputs == 1:
        np.savez_compressed(args.output_fp, prediction)
    else:
        np.savez_compressed(args.output_fp, *prediction)


if __name__ == "__main__":
    main()
