## NOT IMPLEMENTED

"""
Calculate ISM shuffle scores.
"""

import argparse
import logging
import os

import numpy as np
import pyfastx
import tqdm
from scipy.spatial.distance import cdist

import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import clipnet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta_fp", type=str, help="Fasta file path.")
    parser.add_argument(
        "score_fp", type=str, help="Where to write ISM shuffle results."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ensemble_models/",
        help="directory to load models from.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Index of GPU to use (starting from 0). If not invoked, uses CPU.",
    )
    parser.add_argument(
        "--n_shuffles",
        type=int,
        default=5,
        help="Number of shuffles/mutations to perform for each position.",
    )
    parser.add_argument(
        "--mut_size", type=int, default=10, help="Size of mutations to use."
    )
    parser.add_argument(
        "--edge_padding",
        type=int,
        default=50,
        help="Number of positions from edge that we'll skip mutating on.",
    )
    parser.add_argument(
        "--corr_pseudocount",
        type=float,
        default=1e-6,
        help="Pseudocount for correlation calculation.",
    )
    parser.add_argument(
        "--log_quantity_pseudocount",
        type=float,
        default=1e-3,
        help="Pseudocount for log quantity calculation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=617,
        help="Random seed for generating mutations.",
    )
    parser.add_argument(
        "--silence",
        action="store_true",
        help="Disables progress bars and other non-essential print statements.",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Load models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    nn = (
        clipnet.CLIPNET(n_gpus=1, use_specific_gpu=args.gpu)
        if args.gpu is not None
        else clipnet.CLIPNET(n_gpus=0)
    )
    ensemble = nn.construct_ensemble(args.model_dir, silence=args.silence)

    # Load sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sequences = pyfastx.Fasta(args.fasta_fp)
    seqs_twohot = utils.get_twohot_fasta_sequences(args.fasta_fp, silence=args.silence)

    # Calculate ISM shuffle scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    wt_pred = ensemble.predict(seqs_twohot, batch_size=256, verbose=0)
    corr_scores = []
    log_quantity_scores = []
    for i in tqdm.tqdm(
        range(len(sequences)),
        desc="Calculating ISM shuffle scores",
        disable=args.silence,
    ):
        corr_score = []
        log_quantity_score = []
        rec = sequences[i]
        positions = range(args.edge_padding, len(rec.seq) - args.edge_padding)
        for shuffle in range(args.n_shuffles):
            mutated_seqs = []
            for pos in positions:
                mut = utils.kshuffle(rec.seq, random_seed=args.seed)[0][: args.mut_size]
                mutated_seq = (
                    rec.seq[0 : pos - int(len(mut) / 2)]
                    + mut
                    + rec.seq[pos + int(len(mut) / 2) :]
                )
                mutated_seqs.append(mutated_seq)
            mut_pred = ensemble.predict(
                np.array([utils.TwoHotDNA(seq).twohot for seq in mutated_seqs]),
                batch_size=256,
                verbose=0,
            )
            mut_corr = (
                1
                - cdist(
                    np.array(
                        [wt_pred[0][i, :]] * len(positions)
                        + np.random.normal(
                            0, args.corr_pseudocount, mut_pred[0].shape[1]
                        )
                    ),
                    mut_pred[0]
                    + np.random.normal(0, args.corr_pseudocount, mut_pred[0].shape[1]),
                    metric="correlation",
                )[0, :]
            )
            mut_log_quantity = np.log10(
                mut_pred[1] + args.log_quantity_pseudocount
            ) - np.log10(wt_pred[1][i] + args.log_quantity_pseudocount)
            corr_score.append(mut_corr)
            log_quantity_score.append(mut_log_quantity)
        corr_scores.append(np.array(corr_score).mean(axis=0))
        log_quantity_scores.append(np.array(log_quantity_score).mean(axis=0))

    # Save scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    np.savez_compressed(
        args.score_fp,
        corr_ism_shuffle=np.array(corr_scores),
        log_quantity_ism_shuffle=np.array(log_quantity_scores),
    )


if __name__ == "__main__":
    main()
