#!/usr/bin/env python3

"""
This script predicts on a set of data using a fitted model from clipnet.py.
"""

import argparse
import logging
import os

import h5py

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import clipnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fasta_fp",
        type=str,
        help="Input (fasta) file. For individualized genome sequences, \
            heterozygous positions should be represented using IUPAC ambiguity codes.",
    )
    parser.add_argument("output_fp", type=str, help="Output (hdf5) file path.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ensemble_models/",
        help="Directory to load models from",
    )
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        help="Reverse complements the input data.",
    )
    parser.add_argument("--gpu", action="store_true", help="enable GPU.")
    parser.add_argument(
        "--use_specific_gpu",
        type=int,
        default=0,
        help="Index of GPU to use (starting from 0). Does nothing if --gpu is not set.",
    )
    parser.add_argument(
        "--silence",
        action="store_true",
        help="Disables progress bars and other non-essential print statements.",
    )
    args = parser.parse_args()

    nn = (
        clipnet.CLIPNET(n_gpus=1, use_specific_gpu=args.use_specific_gpu)
        if args.gpu
        else clipnet.CLIPNET(n_gpus=0)
    )
    ensemble_predictions = nn.predict_on_fasta(
        model_fp=args.model_dir,
        fasta_fp=args.fasta_fp,
        reverse_complement=args.reverse_complement,
        low_mem=True,
        silence=args.silence,
    )
    with h5py.File(args.output_fp, "w") as hf:
        hf.create_dataset("track", data=ensemble_predictions[0], compression="gzip")
        hf.create_dataset("quantity", data=ensemble_predictions[1], compression="gzip")


if __name__ == "__main__":
    main()
