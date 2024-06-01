#!/usr/bin/env python3

"""
This script fits a NN model using clipnet.py. It requires a NN architecture file, which
must contain a function named construct_nn that returns a tf.keras.models.Model object.
It also requires a dataset_params.py file which specifies parameters and file paths
associated with the dataset of interest.
"""

import argparse
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import clipnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=str, help="directory to save models to")
    parser.add_argument(
        "--prefix",
        type=str,
        default="rnn_v10",
        help="the prefix of the nn architecture file. Must contain a construct_nn \
            method, an opt hash that contains all the optimizer hyperparameters, and a \
            compile_params hash that specifies what loss and metrics are reported. \
            Models will be saved under this prefix.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="resume training from this model.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Index of GPU to use (starting from 0). If not invoked, uses CPU.",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use. If not invoked, uses CPU.",
    )
    args = parser.parse_args()

    if args.n_gpus > 1:
        nn = clipnet.CLIPNET(n_gpus=args.n_gpus, prefix=args.prefix)
    else:
        nn = (
            clipnet.CLIPNET(n_gpus=1, use_specific_gpu=args.gpu, prefix=args.prefix)
            if args.gpu is not None
            else clipnet.CLIPNET(n_gpus=0, prefix=args.prefix)
        )
    nn.fit(model_dir=args.model_dir, resume_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()
