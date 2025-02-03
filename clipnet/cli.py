# CLIPNET CLI
# Adam He <adamyhe@gmail.com>

"""
Wrapper script to calculate attributions and predictions for CLIPNET models
"""

import argparse
import glob
import logging
import os

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)

import tensorflow as tf

from . import attribute, clipnet

_help = """
The following commands are available:
    predict         Calculate predictions for a CLIPNET model
    attribute       Calculate DeepLIFT/SHAP attributions for a CLIPNET model
The following commands are planned but are
    ism_shuffle     Calculate ISM shuffle scores for a CLIPNET model
    epistasis       Calculate Deep Feature Interaction Maps for a CLIPNET model
"""


def cli():
    # DUMMY PARSER FOR COMMON PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_parent = argparse.ArgumentParser(add_help=False)
    parser_parent.add_argument(
        "-f",
        "--fa_fp",
        type=str,
        required=True,
        help="Path to uncompressed fasta file. Can be either a full genome "
        "(in which case a bed file of regions must be provided) or a preprocessed "
        "file of sequences of length 1000.",
    )
    parser_parent.add_argument(
        "-o",
        "--output_fp",
        type=str,
        required=True,
        help="Path to save the main output file. Generally should be a .npz file.",
    )
    parser_parent.add_argument(
        "-m",
        "--model_fp",
        type=str,
        required=True,
        help="Path to directory containing model files or file path to single model file. "
        "If a directory is provided, will calculate predictions or attributions as the "
        "average across all models in the directory.",
    )
    parser_parent.add_argument(
        "-b",
        "--bed_fp",
        type=str,
        default=None,
        help="Path to bed file of regions to calculate predictions/attributions for. "
        "If not provided, will assume that fa_fname contains preprocessed sequences of "
        "length 1000.",
    )
    parser_parent.add_argument(
        "-n",
        "--n_outputs",
        type=int,
        default=2,
        help="Number of outputs for the model. Default is 2 for standard CLIPNET models "
        "and 1 for the single scalar output models (e.g. PauseNet, OrientNet, InitNet).",
    )
    parser_parent.add_argument(
        "-c",
        "--chroms",
        type=str,
        nargs="+",
        default=None,
        help="Chromosomes to calculate predictions or attributions for. Defaults to all. "
        "Only used if bed_fname is provided, as we can use those to filter chroms.",
    )
    parser_parent.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size to control VRAM usage. Defaults to 16.",
    )
    parser_parent.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to use. If -1, uses CPU.",
    )
    parser_parent.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to print progress bars."
    )

    # MAIN PARSER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help="The following commands are available:", required=True, dest="cmd"
    )

    # PREDICT PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_predict = subparsers.add_parser(
        "predict",
        help="Calculate predictions for a given set of regions.",
        parents=[parser_parent],
    )
    parser_predict.add_argument(
        "-s",
        "--signal_fname",
        type=str,
        nargs=2,
        default=None,
        help="Signal files containing experimental data to benchmark model "
        "predictions against. Expected order is [plus_bigWig, minus_bigWig]. "
        "If not provided, will not calculate performance metrics.",
    )

    # ATTRIBUTE PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_attribute = subparsers.add_parser(
        "attribute",
        help="Calculate attributions for a given set of regions.",
        parents=[parser_parent],
    )
    parser_attribute.add_argument(
        "-m",
        "--mode",
        type=str,
        default="quantity",
        choices={"quantity", "profile"},
        help="The type of attribution to calculate.",
    )
    parser_attribute.add_argument(
        "-s",
        "--save_ohe",
        type=str,
        default=None,
        help="Where to save OHE of sequences. Defaults to not saving.",
    )
    parser_attribute.add_argument(
        "-y",
        "--hyp_attr_fp",
        type=str,
        default=None,
        help="Where to save hypothetical attributions. Defaults to not saving.",
    )
    parser_attribute.add_argument(
        "-n",
        "--n_shuffles",
        type=int,
        default=100,
        help="Number of dinucleotide shuffles for DeepLIFT/SHAP. Defaults to 100.",
    )
    parser_attribute.add_argument(
        "-r",
        "--random_state",
        type=int,
        default=47,
        help="Random seed. Defaults to 47.",
    )

    # ISM SHUFFLE PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_ism_shuffle = subparsers.add_parser(
        "ism_shuffle",
        help="Calculate ISM shuffle scores for a given set of regions.",
        parents=[parser_parent],
    )

    # EPISTASIS PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_epistasis = subparsers.add_parser(
        "epistasis",
        help="Calculate Deep Feature Interaction Maps for a given set of regions.",
        parents=[parser_parent],
    )

    args = parser.parse_args()

    # MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Set correct device
    nn = (
        clipnet.CLIPNET(n_gpus=1, use_specific_gpu=args.gpu)
        if args.gpu is not None
        else clipnet.CLIPNET(n_gpus=0)
    )

    # PREDICT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if args.cmd == "predict":
        ensemble_predictions = nn.predict_on_fasta(
            model_fp=args.model_fp,
            fasta_fp=args.fasta_fp,
            outputs=args.n_outputs,
            reverse_complement=args.reverse_complement,
            low_mem=True,
            silence=not args.verbose,
        )
        if args.outputs == 1:
            np.savez_compressed(args.output_fp, ensemble_predictions)
        else:
            np.savez_compressed(args.output_fp, *ensemble_predictions)

    # ATTRIBUTE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    elif args.cmd == "attribute":
        # Disable TF32 to reduce potential low precision issues
        tf.config.experimental.enable_tensor_float_32_execution(False)

        # Load data
        seqs_to_explain, twohot_background = attribute.load_seqs(
            args.fasta_fp, n_subset=args.n_subset, seed=args.seed
        )

        # Define contribution function
        if args.n_outputs == 1:
            contrib = attribute.scalar_contrib
        if args.mode == "quantity":
            contrib = attribute.quantity_contrib
        elif args.mode == "profile":
            contrib = attribute.profile_contrib

        # Create explainers
        if os.path.isdir(args.model_fp):
            model_names = list(glob.glob(os.path.join(args.model_dir, "*.h5")))
        else:
            model_names = [args.model_fp]
        explainers = attribute.create_explainers(
            model_names,
            twohot_background,
            contrib,
            len(model_names) == 1 or args.silence,
        )

        # Convert twohot to onehot and save
        if args.save_ohe is not None:
            np.savez_compressed(
                args.save_ohe, (seqs_to_explain / 2).astype(int).swapaxes(1, 2)
            )

        # Calculate attributions
        explanations, hyp_explanations = attribute.calculate_scores(
            explainers,
            seqs_to_explain,
            batch_size=args.batch_size,
            silence=args.silence,
            check_additivity=not args.skip_check_additivity,
        )

        # Save
        np.savez_compressed(args.score_fp, explanations.swapaxes(1, 2))
        # Save hypothetical attributions
        if args.hyp_attr_fp is not None:
            np.savez_compressed(args.hyp_attr_fp, hyp_explanations.swapaxes(1, 2))

    # ISM SHUFFLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    elif args.cmd == "ism_shuffle":
        pass

    # EPISTASIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    elif args.cmd == "epistasis":
        pass

    # INVALID ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    cli()
