## NOT IMPLEMENTED

"""
Calculate ISM shuffle scores.
"""

import numpy as np
import tqdm
from scipy.spatial.distance import cdist
from silence_tensorflow import silence_tensorflow

from . import shuffle, utils

silence_tensorflow()


def ism_shuffle(
    model,
    sequences,
    correction=1,
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
        for j in range(n_shuffles):
            mutated_seqs = []
            for pos in positions:
                mut = shuffle.kshuffle(rec.seq)[0][:mut_size]
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
            mut_logfc = correction * np.log(
                mut_pred[1] + logfc_pseudocount
            ) - correction * np.log(wt_pred[1][i] + logfc_pseudocount)
            corr_score.append(mut_corr)
            logfc_score.append(mut_logfc)
        corr_scores.append(np.array(corr_score).mean(axis=0))
        logfc_scores.append(np.array(logfc_score).mean(axis=0))

    return np.array(corr_scores), np.array(logfc_scores)
