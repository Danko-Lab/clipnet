#!/usr/bin/env python3

"""
This script predicts on a set of data using a fitted model from clipnet.py.
"""

import argparse

import h5py

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
        "--low_mem", action="store_true", help="Predict in batches to fit in VRAM."
    )
    args = parser.parse_args()

    nn = clipnet.CLIPNET(n_gpus=1) if args.gpu else clipnet.CLIPNET(n_gpus=0)
    ensemble_predictions = nn.predict_ensemble(
        model_dir=args.model_dir,
        fasta_fp=args.fasta_fp,
        reverse_complement=args.reverse_complement,
        low_mem=args.low_mem,
    )
    with h5py.File(args.output_fp, "w") as hf:
        hf.create_dataset("track", data=ensemble_predictions[0], compression="gzip")
        hf.create_dataset("quantity", data=ensemble_predictions[1], compression="gzip")


if __name__ == "__main__":
    main()
