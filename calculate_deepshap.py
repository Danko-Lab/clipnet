"""
Calculate contribution scores using shap.DeepExplainer.
"""

import argparse
import gc
import logging
import os

import numpy as np
import pyfastx
import shap
import tqdm

import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf

# This will fix an error message for running tf.__version__==2.5
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
    shap.explainers._deep.deep_tf.passthrough
)
tf.compat.v1.disable_v2_behavior()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta_fp", type=str, help="Fasta file path.")
    parser.add_argument("score_fp", type=str, help="Where to write DeepSHAP scores.")
    parser.add_argument("seq_fp", type=str, help="Where to write onehot sequences.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory to load models from",
    )
    parser.add_argument(
        "--model_fp",
        type=str,
        default=None,
        help="Model file path. Use to calculate for a specific model fold. \
            Selecting this option will override --model_dir.",
    )
    parser.add_argument(
        "--hyp_attr_fp",
        type=str,
        default=None,
        help="Where to write hypothetical attributions.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quantity",
        help="Calculate contrib scores for quantity or profile.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Index of GPU to use (starting from 0). If not invoked, uses CPU.",
    )
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
    np.random.seed(args.seed)

    if args.model_fp is not None and args.model_dir is not None:
        raise ValueError("Cannot specify both --model_fp and --model_dir.")
    if args.model_fp is None and args.model_dir is None:
        raise ValueError("Must specify either --model_fp or --model_dir.")
    if args.mode not in ["quantity", "profile"]:
        raise ValueError("mode must be either 'quantity' or 'profile'.")

    # Load sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    sequences = pyfastx.Fasta(args.fasta_fp)
    seqs_to_explain = utils.get_twohot_fasta_sequences(
        args.fasta_fp, silence=args.silence
    )

    # Perform dinucleotide shuffle on n_subset random sequences
    if args.n_subset > len(sequences):
        print(
            "n_subset (%d) > sequences in the fasta file (%d)."
            % (args.n_subset, len(sequences)),
            "Using all sequences to generate DeepSHAP reference.",
        )
    reference = [
        sequences[i]
        for i in np.random.choice(
            np.array(range(len(sequences))),
            size=min(args.n_subset, len(sequences)),
            replace=False,
        )
    ]
    shuffled_reference = [
        utils.kshuffle(rec.seq, random_seed=np.random.RandomState(args.random_seed))[0]
        for rec in reference
    ]

    # Two-hot encode shuffled sequences
    twohot_reference = np.array(
        [utils.TwoHotDNA(seq).twohot for seq in shuffled_reference]
    )

    # Load model and create explainer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if args.gpu is not None:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if args.gpu >= len(gpus):
            raise IndexError(
                f"Requested GPU index {args.gpu} does not exist ({len(gpus)} total)."
            )
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        gpus = tf.config.list_physical_devices("GPU")
        gpu = gpus[args.gpu]
        print(f"Using GPU {gpu}.")
        tf.config.set_visible_devices(gpu, "GPU")

    if args.model_fp is None:
        models = [
            tf.keras.models.load_model(f"{args.model_dir}/fold_{i}.h5", compile=False)
            for i in tqdm.tqdm(
                range(1, 10), desc="Loading models", disable=args.silence
            )
        ]
    else:
        models = [tf.keras.models.load_model(args.model_fp, compile=False)]
    if args.mode == "quantity":
        contrib = [model.output[1] for model in models]
        check_additivity = not args.skip_check_additivity
    else:
        softmax = tf.keras.layers.Softmax()
        contrib = [
            tf.reduce_mean(
                tf.stop_gradient(softmax(model.output[0])) * model.output[0],
                axis=-1,
                keepdims=True,
            )
            for model in models
        ]
        check_additivity = not args.skip_check_additivity
    explainers = [
        shap.DeepExplainer((model.input, contrib), twohot_reference)
        for (model, contrib) in tqdm.tqdm(
            zip(models, contrib),
            desc="Creating explainers",
            total=len(models),
            disable=args.silence or len(models) == 1,
        )
    ]

    # Calculate scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    hyp_explanations = {i: [] for i in range(len(explainers))}
    batch_size = 256
    for i, explainer in enumerate(explainers):
        desc = "Calculating explanations"
        if len(explainers) > 1:
            desc += f" for model fold {i + 1}"
        for j in tqdm.tqdm(
            range(0, len(seqs_to_explain), batch_size), desc=desc, disable=args.silence
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

    # Save scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Save DeepSHAP scores
    np.savez_compressed(args.score_fp, explanations.swapaxes(1, 2))
    # Convert twohot to onehot and save
    np.savez_compressed(args.seq_fp, (seqs_to_explain / 2).astype(int).swapaxes(1, 2))
    if args.hyp_attr_fp is not None:
        np.savez_compressed(args.hyp_attr_fp, mean_explanations.swapaxes(1, 2))


if __name__ == "__main__":
    main()
