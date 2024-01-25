"""
Calculate profile attribution scores using shap.DeepExplainer.
"""

import argparse
import gc

import numpy as np
import pyfastx
import shap
import tensorflow as tf
import ushuffle

import utils

# This will fix an error message for running tf.__version__==2.5
shap.explainers._deep.deep_tf.op_handlers[
    "AddV2"
] = shap.explainers._deep.deep_tf.passthrough
tf.compat.v1.disable_v2_behavior()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_fp", type=str, help="Model file path.")
    parser.add_argument("fasta_fp", type=str, help="Fasta file path.")
    parser.add_argument("score_fp", type=str, help="Where to write DeepSHAP scores.")
    parser.add_argument(
        "seq_fp", type=int, help="Where to write one-encoding of sequences."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quantity",
        help="Calculate contrib scores for quantity or profile.",
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU.")
    parser.add_argument(
        "--n_subset",
        type=int,
        default=100,
        help="Maximum number of sequences to use as background. \
            Default is 100 to ensure reasonably fast compute on large datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=617,
        help="Random seed for selecting background sequences.",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)

    assert args.mode in [
        "quantity",
        "profile",
    ], "mode must be either quantity or profile."

    # Load sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sequences = pyfastx.Fasta(args.fasta_fp)
    seqs_to_explain = np.array([utils.OneHotDNA(seq).onehot for seq in sequences])

    # Perform dinucleotide shuffle on n_subset random sequences
    if len(sequences) < args.n_subset:
        args.n_subset = len(sequences)
    reference = [
        sequences[i]
        for i in np.random.choice(
            np.array(range(len(sequences))), size=args.n_subset, replace=False
        )
    ]
    shuffled_reference = [ushuffle.shuffle(rec.seq, 1000, 2) for rec in reference]

    # One-hot encode shuffled sequences
    onehot_reference = np.array(
        [utils.OneHotDNA(seq).onehot for seq in shuffled_reference]
    )

    # Load model and create explainer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[-1], "GPU")
    else:
        tf.config.set_visible_devices([], "GPU")

    model = tf.keras.models.load_model(args.model_fp, compile=False)
    if args.mode == "quantity":
        contrib = model.output[1]
    else:
        contrib = tf.reduce_mean(
            tf.stop_gradient(tf.nn.softmax(model.output[0], axis=-1)) * model.output[0],
            axis=-1,
            keepdims=True,
        )
    explainer = shap.DeepExplainer((model.input, contrib), onehot_reference)

    # Calculate scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    raw_explanations = []
    batch_size = 256
    for i in range(0, len(seqs_to_explain), batch_size):
        print(f"Calculating scores for input sequences {i} to {i+batch_size}")
        raw_explanations.append(
            explainer.shap_values(seqs_to_explain[i : i + batch_size])
        )
        gc.collect()

    concat_exp = np.concatenate([exp for exp in raw_explanations], axis=1).sum(axis=0)
    scaled_explanations = concat_exp * seqs_to_explain

    # Save scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print(f"Finished calculating scores for model: {str(args.model_fp)}")

    np.savez_compressed(args.score_fp, scaled_explanations.swapaxes(1, 2))
    np.savez_compressed(args.seq_fp, (seqs_to_explain / 2).astype(int).swapaxes(1, 2))


if __name__ == "__main__":
    main()
