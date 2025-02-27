## NOT IMPLEMENTED

"""
Calculate contribution scores using shap.DeepExplainer.
"""

import argparse
import gc

import numpy as np
import shap
import tensorflow as tf

import .utils

# This will fix an error message for running tf.__version__==2.5
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
    shap.explainers._deep.deep_tf.passthrough
)
tf.compat.v1.disable_v2_behavior()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("core_promoter_seq", type=str, help="")
    parser.add_argument("motif", type=str, help="Where to write DeepSHAP scores.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ensemble_models",
        help="Directory to load models from",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quantity",
        help="Calculate interaction strength using quantity or DFIM",
    )
    parser.add_argument("--gpu", action="store_true", help="Enable GPU.")
    parser.add_argument(
        "--use_specific_gpu",
        type=int,
        default=0,
        help="Index of GPU to use (starting from 0). Does nothing if --gpu is not set.",
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
        default=617,
        help="Random seed for selecting background sequences.",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)

    assert args.mode in [
        "quantity",
        "dfim",
    ], "mode must be either quantity or dfim."

    # Load sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    concat_ref = "".join(shuffled_ref)
    embedded_promoter = (
        ushuffle.shuffle(
            concat_ref, int(np.floor((1000 - len(core_promoter_seq)) / 2)), 2
        )
        + core_promoter_seq
        + ushuffle.shuffle(
            concat_ref, int(np.ceil((1000 - len(core_promoter_seq)) / 2)), 2
        )
    )
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
    shuffled_reference = [
        utils.kshuffle(rec.seq, random_seed=args.seed)[0] for rec in reference
    ]

    # One-hot encode shuffled sequences
    onehot_reference = np.array(
        [utils.OneHotDNA(seq).onehot for seq in shuffled_reference]
    )

    # Load model and create explainer ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if args.gpu:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[args.use_specific_gpu], "GPU")
    else:
        tf.config.set_visible_devices([], "GPU")

    if args.model_fp is None:
        models = [
            tf.keras.models.load_model(f"{args.model_dir}/fold_{i}.h5", compile=False)
            for i in range(1, 10)
        ]
    else:
        models = [tf.keras.models.load_model(args.model_fp, compile=False)]
    if args.mode == "quantity":
        contrib = [model.output[1] for model in models]
    else:
        contrib = [
            tf.reduce_mean(
                tf.stop_gradient(tf.nn.softmax(model.output[0], axis=-1))
                * model.output[0],
                axis=-1,
                keepdims=True,
            )
            for model in models
        ]
    explainers = [
        shap.DeepExplainer((model.input, contrib), onehot_reference)
        for (model, contrib) in zip(models, contrib)
    ]

    # Calculate scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    raw_explanations = {i: [] for i in range(len(explainers))}
    batch_size = 256
    for i in range(len(explainers)):
        for j in range(0, len(seqs_to_explain), batch_size):
            shap_values = explainers[i].shap_values(seqs_to_explain[j : j + batch_size])
            raw_explanations[i].append(shap_values)
            gc.collect()

    concat_explanations = []
    for k in raw_explanations.keys():
        concat_explanations.append(
            np.concatenate([exp for exp in raw_explanations[k]], axis=1).sum(axis=0)
        )

    mean_explanations = np.array(concat_explanations).mean(axis=0)
    scaled_explanations = mean_explanations * seqs_to_explain

    # Save scores ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    np.savez_compressed(args.score_fp, scaled_explanations.swapaxes(1, 2))
    np.savez_compressed(args.seq_fp, (seqs_to_explain / 2).astype(int).swapaxes(1, 2))


if __name__ == "__main__":
    main()
