"""
Calculate profile attribution scores using shap.DeepExplainer.
"""

import gc
import os
import sys
from pathlib import Path

import numpy as np
import pyfastx
import shap
import tensorflow as tf
import ushuffle

import utils

model_fold = sys.argv[1]

# This will fix an error message for running tf.__version__==2.5
shap.explainers._deep.deep_tf.op_handlers[
    "AddV2"
] = shap.explainers._deep.deep_tf.passthrough
tf.compat.v1.disable_v2_behavior()

np.random.seed(617)
n_subset = 100

# Load sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

motif = "all"

fasta_fp = os.path.join(
    "/home2/ayh8/",
    "data/gse110638/tfbs_sampling/",
    f"{motif}/{motif}_tss_windows_reference_seq.fna.gz",
)
sequences = pyfastx.Fasta(fasta_fp)
seqs_to_explain = utils.get_onehot_fasta_sequences(fasta_fp)

# Perform dinucleotide shuffle on 100 random sequences
reference = [
    sequences[i]
    for i in np.random.choice(
        np.array(range(len(sequences))), size=n_subset, replace=False
    )
]
shuffled_reference = [ushuffle.shuffle(rec.seq, 1000, 2) for rec in reference]

# One-hot encode shuffled sequences
onehot_reference = np.array([utils.OneHotDNA(seq).onehot for seq in shuffled_reference])


# Load model and create explainer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[-1], "GPU")

model_fp = Path(
    f"/home2/ayh8/ensemble_models/fold_{model_fold}.h5",
)
model = tf.keras.models.load_model(model_fp, compile=False)
profile_contrib = tf.reduce_mean(
    tf.stop_gradient(tf.nn.softmax(model.output[0], axis=-1)) * model.output[0],
    axis=-1,
    keepdims=True,
)
explainer = shap.DeepExplainer((model.input, profile_contrib), onehot_reference)


# Calculate scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

raw_explanations = []
batch_size = 256
for i in range(0, len(seqs_to_explain), batch_size):
    print(f"Calculating scores for {i} to {i+batch_size}")
    raw_explanations.append(explainer.shap_values(seqs_to_explain[i : i + batch_size]))
    gc.collect()

concat_exp = np.concatenate([exp for exp in raw_explanations], axis=1).sum(axis=0)
scaled_explanations = concat_exp * seqs_to_explain


# Save scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print(f"Finished calculating scores for model: {str(model_fp)}")

out_dir = Path("/home2/ayh8/attribution_scores/", f"{motif}")
out_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_dir.joinpath(f"model_fold_{model_fold}_{motif}_profile.npz"),
    scaled_explanations.swapaxes(1, 2),
)

np.savez_compressed(
    out_dir.joinpath(f"{motif}_seqs_onehot.npz"),
    (seqs_to_explain / 2).astype(int).swapaxes(1, 2),
)
