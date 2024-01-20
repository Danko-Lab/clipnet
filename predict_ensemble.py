#!/usr/bin/env python3

"""
This script predicts on a set of data using a fitted model from clipnet.py.
"""

import argparse

import h5py

import clipnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta_fp", type=str, help="fasta file.")
    parser.add_argument("output_fp", type=str, help="output file path.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ensemble_models/",
        help="directory to load models from",
    )
    parser.add_argument(
        "--reverse_complement",
        action="store_true",
        help="reverse complements the input data.",
    )
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus to use.")
    parser.add_argument(
        "--use_specific_gpu",
        type=int,
        default=None,
        help="If n_gpus==1, allows choice of specific gpu.",
    )
    parser.add_argument(
        "--low_mem", action="store_true", help="Use smaller batch size to fit in VRAM."
    )
    args = parser.parse_args()

    nn = clipnet.CLIPNET(
        n_gpus=args.n_gpus,
        use_specific_gpu=args.use_specific_gpu,
    )
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
