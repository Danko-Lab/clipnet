#!/usr/bin/env python3

"""
This script predicts on a set of data using a fitted model from clipnet.py.
"""

import argparse

import h5py

import clipnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_fp", type=str, help="model file path.")
    parser.add_argument("fasta_fp", type=str, help="fasta file.")
    parser.add_argument("output_fp", type=str, help="output file path.")
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        help="reverse complements the input data.",
    )
    parser.add_argument("--gpu", action="store_true", help="enable GPU.")
    parser.add_argument(
        "--use_specific_gpu",
        type=int,
        default=0,
        help="Index of GPU to use (starting from 0). Does nothing if --gpu is not set.",
    )
    parser.add_argument(
        "--low_mem", action="store_true", help="Use smaller batch size to fit in VRAM."
    )
    args = parser.parse_args()

    nn = (
        clipnet.CLIPNET(n_gpus=1, use_specific_gpu=args.use_specific_gpu)
        if args.gpu
        else clipnet.CLIPNET(n_gpus=0)
    )

    prediction = nn.predict_on_fasta(
        model_fp=args.model_fp,
        fasta_fp=args.fasta_fp,
        reverse_complement=args.reverse_complement,
        low_mem=args.low_mem,
    )
    with h5py.File(args.output_fp, "w") as hf:
        hf.create_dataset("track", data=prediction[0], compression="gzip")
        hf.create_dataset("quantity", data=prediction[1], compression="gzip")


if __name__ == "__main__":
    main()
