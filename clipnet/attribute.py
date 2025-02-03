# attribute.py
# Adam He <adamyhe@gmail.com>

"""
Functions for calculating attribution scores using shap.DeepExplainer
"""

import gc

import numpy as np
import pyfastx
import tqdm
from silence_tensorflow import silence_tensorflow

from . import shuffle, utils

silence_tensorflow()
import shap
import tensorflow as tf

# This may be needed to fix certain version incompatibilities
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
    shap.explainers._deep.deep_tf.passthrough
)
tf.compat.v1.disable_v2_behavior()


def scalar_contrib(model):
    return model.output


def quantity_contrib(model):
    return model.output[1]


def profile_contrib_(model):
    """
    Original implementation used in CLIPNET. Deprecated, but kept here
    for documentation.
    """
    softmax = tf.keras.layers.Softmax()
    contrib = tf.reduce_mean(
        tf.stop_gradient(softmax(model.output[0])) * model.output[0],
        axis=-1,
        keepdims=True,
    )
    return contrib


def profile_contrib(model):
    """
    Adapted from bpnetlite.bpnet.ProfileWrapper
    """
    softmax = tf.keras.layers.Softmax()
    avg = tf.keras.layers.Average(axis=-1, keepdims=True)
    add = tf.keras.layers.Add(axis=-1, keepdims=True)
    logits = model.output[0] - avg(model.output[0])
    contrib = add(softmax(model.output[0]) * logits)
    return contrib


def load_seqs(
    fasta_fp, return_twohot_explains=True, background_fp=None, n_subset=100, seed=None
):
    np.random.seed(seed)
    seqs_to_explain = pyfastx.Fasta(fasta_fp)
    background_seqs = (
        seqs_to_explain if background_fp is None else pyfastx.Fasta(background_fp)
    )
    reference = [
        background_seqs[i]
        for i in np.random.choice(
            np.array(range(len(background_seqs))), size=n_subset, replace=True
        )
    ]
    shuffled_reference = [
        shuffle.kshuffle(rec.seq, random_seed=seed)[0] for rec in reference
    ]
    twohot_background = np.array(
        [utils.TwoHotDNA(seq).twohot for seq in shuffled_reference]
    )
    if return_twohot_explains:
        seqs_to_explain = np.array(
            [utils.TwoHotDNA(seq).twohot for seq in seqs_to_explain]
        )
    return seqs_to_explain, twohot_background


def create_explainers(model_fps, twohot_background, contrib, silence=False):
    models = [
        tf.keras.models.load_model(fp, compile=False)
        for fp in tqdm.tqdm(model_fps, desc="Loading models")
    ]
    explainers = []
    for model in tqdm.tqdm(models, desc="Creating explainers", disable=silence):
        explainers.append(
            shap.DeepExplainer((model.input, contrib(model)), twohot_background)
        )
    return explainers


def calculate_scores(
    explainers, seqs_to_explain, batch_size=256, check_additivity=True, silence=False
):
    hyp_explanations = {i: [] for i in range(len(explainers))}
    for i, explainer in enumerate(explainers):
        desc = "Calculating explanations"
        if len(explainers) > 1:
            desc += f" for model fold {i + 1}"
        for j in tqdm.tqdm(
            range(0, len(seqs_to_explain), batch_size), desc=desc, disable=silence
        ):
            shap_values = explainer.shap_values(
                seqs_to_explain[j : j + batch_size], check_additivity=check_additivity
            )
            hyp_explanations[i].append(shap_values)
            gc.collect()

    concat_explanations = [
        np.concatenate([exp[0] for exp in hyp_explanations[k]], axis=0)
        for k in hyp_explanations.keys()
    ]

    if len(explainers) > 1:
        mean_explanations = np.array(concat_explanations).mean(axis=0)
    else:
        mean_explanations = concat_explanations[0]
    explanations = mean_explanations * seqs_to_explain
    return explanations, mean_explanations
