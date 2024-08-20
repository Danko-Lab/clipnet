"""
Calculate Deep Feature Interaction Maps (DFIM) for a bunch of examples.
"""

import argparse
import glob
import logging
import os

import numpy as np
import tqdm

import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import shap
import tensorflow as tf

import clipnet
from calculate_deepshap import (
    create_explainers,
    load_seqs,
    profile_contrib,
    quantity_contrib,
)

# This will fix an error message for running tf.__version__==2.5
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
    shap.explainers._deep.deep_tf.passthrough
)
tf.compat.v1.disable_v2_behavior()


def calculate_dfim(explainers, rec, start, stop, check_additivity=True, silence=False):
    major_seq = rec.seq
    major_twohot = np.expand_dims(utils.TwoHotDNA(major_seq).twohot, axis=0)
    dfim_range = list(range(start, stop))
    major_shap = np.array(
        [
            explainer.shap_values(major_twohot, check_additivity=check_additivity)[0]
            for explainer in explainers
        ]
    ).mean(axis=0)
    mutations_per_pos = [utils.get_mut_bases(major_seq[i]) for i in dfim_range]
    dfim = []
    for i in tqdm.trange(
        len(mutations_per_pos),
        desc=f"Calculating DFIM for {rec.name} (pos {start}-{stop})",
        disable=silence,
    ):
        muts = mutations_per_pos[i]
        mut_seqs = [
            major_seq[: dfim_range[i]] + mut + major_seq[dfim_range[i] + 1 :]
            for mut in muts
        ]
        mut_twohot = np.array([utils.TwoHotDNA(mut_seq).twohot for mut_seq in mut_seqs])
        mut_shap = np.array(
            [
                explainer.shap_values(mut_twohot, check_additivity=check_additivity)[0]
                for explainer in explainers
            ]
        ).mean(axis=0)
        fis = (
            np.abs(major_shap * major_twohot / 2 - mut_shap * mut_twohot / 2)
            .sum(axis=2)
            .max(axis=0)[start:stop]
        )
        dfim.append(fis)
    return np.array(dfim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta_fp", type=str, help="Fasta file path.")
    parser.add_argument("score_fp", type=str, help="Where to write DFIM scores.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ensemble_models/",
        help="Directory to load models from",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quantity",
        help="Calculate contrib scores for quantity or profile.",
    )
    parser.add_argument(
        "--start", type=int, default=400, help="Start position for calculating DFIM."
    )
    parser.add_argument(
        "--stop", type=int, default=600, help="Stop position for calculating DFIM."
    )
    parser.add_argument(
        "--background_fp",
        type=str,
        default=None,
        help="Background sequences (if None, will select from main seqs).",
    )
    parser.add_argument(
        "--n_subset",
        type=int,
        default=100,
        help="Maximum number of sequences to use as background. \
            Default is 100 to ensure reasonably fast compute on large datasets.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Index of GPU to use (starting from 0). If not invoked, uses CPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for selecting background sequences.",
    )
    parser.add_argument(
        "--silence",
        action="store_true",
        help="Disables progress bars and other non-essential print statements.",
    )
    parser.add_argument(
        "--skip_check_additivity",
        action="store_true",
        help="Disables check for additivity of shap results.",
    )
    args = parser.parse_args()

    # Check arguments ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if args.model_fp is None and args.model_dir is None:
        raise ValueError("Must specify either --model_fp or --model_dir.")
    if args.mode == "quantity":
        contrib = quantity_contrib
    elif args.mode == "profile":
        contrib = profile_contrib
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'quantity' or 'profile'.")

    # Load sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    seqs_to_explain, twohot_background = load_seqs(
        args.fasta_fp, False, args.background_fp, args.n_subset, args.seed
    )

    # Create explainers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    nn = clipnet.CLIPNET(n_gpus=1, use_specific_gpu=args.gpu)
    model_fps = list(glob.glob(os.path.join(args.model_dir, "*.h5")))
    explainers = create_explainers(model_fps, twohot_background, contrib, args.silence)

    # Calculate DFIM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dfims = {
        rec.name: calculate_dfim(
            explainers,
            rec,
            args.start,
            args.stop,
            check_additivity=not args.skip_check_additivity,
            silence=args.silence,
        )
        for rec in seqs_to_explain
    }
    np.savez(args.score_fp, **dfims)


if __name__ == "__main__":
    main()
