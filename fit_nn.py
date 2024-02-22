#!/usr/bin/env python3

"""
This script fits a NN model using clipnet.py. It requires a NN architecture file, which
must contain a function named construct_nn that returns a tf.keras.models.Model object.
It also requires a dataset_params.py file which specifies parameters and file paths
associated with the dataset of interest.
"""

import argparse

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
    parser.add_argument("--n_gpus", type=int, default=1, help="number of gpus to use.")
    parser.add_argument(
        "--use_specific_gpu",
        type=int,
        default=None,
        help="If n_gpus==1, allows choice of specific gpu.",
    )
    args = parser.parse_args()
    resume = args.resume_checkpoint
    model_dir = args.model_dir
    args_vars = vars(args)
    del args_vars["resume_checkpoint"]
    del args_vars["model_dir"]

    nn = clipnet.CLIPNET(**args_vars)
    nn.fit(model_dir=model_dir, resume_checkpoint=resume)


if __name__ == "__main__":
    main()
