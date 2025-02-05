## NOT IMPLEMENTED

"""
Calculate ISM shuffle scores.
"""

import argparse

import numpy as np
import pyfastx
import tqdm
from scipy.spatial.distance import cdist
from silence_tensorflow import silence_tensorflow

from . import utils

silence_tensorflow()
from . import clipnet


def ism_shuffle(
    model,
    sequences,
    mut_size=10,
    n_shuffles=5,
    edge_padding=50,
    corr_pseudocount=1e-6,
    logfc_pseudocount=1e-6,
    batch_size=256,
    verbose=False,
):
    seqs_twohot = np.array([utils.TwoHotDNA(seq).twohot for seq in sequences])
    wt_pred = model.predict(seqs_twohot, batch_size=batch_size, verbose=verbose)
    corr_scores = []
    logfc_scores = []
    for i in tqdm.trange(
        len(sequences), desc="Calculating ISM shuffle scores", disable=not verbose
    ):
        corr_score = []
        logfc_score = []
        rec = sequences[i]
        positions = range(edge_padding, len(rec.seq) - edge_padding)
        for shuffle in range(n_shuffles):
            mutated_seqs = []
            for pos in positions:
                mut = utils.kshuffle(rec.seq)[0][:mut_size]
                mutated_seq = (
                    rec.seq[0 : pos - int(len(mut) / 2)]
                    + mut
                    + rec.seq[pos + int(len(mut) / 2) :]
                )
                mutated_seqs.append(mutated_seq)
            mut_pred = model.predict(
                np.array([utils.TwoHotDNA(seq).twohot for seq in mutated_seqs]),
                batch_size=256,
                verbose=0,
            )
            mut_corr = (
                1
                - cdist(
                    np.array(
                        [wt_pred[0][i, :]] * len(positions)
                        + np.random.normal(0, corr_pseudocount, mut_pred[0].shape[1])
                    ),
                    mut_pred[0]
                    + np.random.normal(0, corr_pseudocount, mut_pred[0].shape[1]),
                    metric="correlation",
                )[0, :]
            )
            mut_logfc = np.log(mut_pred[1] + logfc_pseudocount) - np.log(
                wt_pred[1][i] + logfc_pseudocount
            )
            corr_score.append(mut_corr)
            logfc_score.append(mut_logfc)
        corr_scores.append(np.array(corr_score).mean(axis=0))
        logfc_scores.append(np.array(logfc_score).mean(axis=0))

    return np.array(corr_scores), np.array(logfc_scores)


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
        "--logfc_pseudocount",
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

    # Save scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    np.savez_compressed(
        args.score_fp,
        corr_ism_shuffle=np.array(corr_scores),
        logfc_ism_shuffle=np.array(logfc_scores),
    )


if __name__ == "__main__":
    main()
