# CLIPNET CLI
# Adam He <adamyhe@gmail.com>

"""
CLI for calculating attributions and predictions with CLIPNET models.
"""

import argparse
import glob
import os

import numpy as np
import pyfastx
import seqlogo
import tqdm
from silence_tensorflow import silence_tensorflow

silence_tensorflow()
import shap
import tensorflow as tf

from . import attribute, clipnet, epistasis, ism_shuffle

# This may be needed to fix certain version incompatibilities
shap.explainers._deep.deep_tf.op_handlers["AddV2"] = (
    shap.explainers._deep.deep_tf.passthrough
)
tf.compat.v1.disable_v2_behavior()
# Needed to register a custom nonlinear operation
shap.explainers._deep.deep_tf.op_handlers["_profile_logit_scaling"] = (
    shap.explainers._deep.deep_tf.nonlinearity_1d(0)
)


_help = """
The following commands are available:
    predict         Calculate predictions for a CLIPNET model
    attribute       Calculate DeepLIFT/SHAP attributions for a CLIPNET model
    ism_shuffle     Calculate ISM shuffle scores for a CLIPNET model
    epistasis       Calculate Deep Feature Interaction Maps for a CLIPNET model
    tss_pwm         Calculate TSS PWMs for a CLIPNET model
The following commands are planned but are not yet implemented.
    activation_maps
    fit
"""


def cli():
    # DUMMY PARSER FOR COMMON PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_parent = argparse.ArgumentParser(add_help=False)
    parser_parent.add_argument(
        "-f",
        "--fasta_fp",
        type=str,
        required=True,
        help="Path to fasta file containing preprocessed sequences of length 1000.",
    )
    parser_parent.add_argument(
        "-o",
        "--output_fp",
        type=str,
        required=True,
        help="Path (.npz) to save the main output file.",
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
        "-n",
        "--n_outputs",
        type=int,
        default=2,
        help="Number of outputs for the model. Default is 2 for standard CLIPNET models "
        "and 1 for the single scalar output models (e.g. PauseNet, OrientNet, InitNet).",
    )
    parser_parent.add_argument(
        "-b",
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
        help="Which GPU to use (default is 0, the first one). If -1, uses CPU.",
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

    # ATTRIBUTE PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_attribute = subparsers.add_parser(
        "attribute",
        help="Calculate attributions for a given set of regions.",
        parents=[parser_parent],
    )
    parser_attribute.add_argument(
        "-a",
        "--attribution_type",
        type=str,
        default="quantity",
        choices={"quantity", "profile"},
        help="The type of attribution to calculate. REQUIRED if n_outputs == 2. "
        "if n_outputs == 1, this does nothing",
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
        "--save_hypothetical",
        action="store_true",
        help="Whether to save hypothetical sequences. If not set, will save actual "
        "contributions, i.e., ohe * attribution. This should be set if you intend to "
        "run tfmodisco on the attributions.",
    )
    parser_attribute.add_argument(
        "-d",
        "--n_dinucleotide_shuffles",
        type=int,
        default=100,
        help="Number of dinucleotide shuffles for DeepLIFT/SHAP background reference. "
        "Defaults to 100.",
    )
    parser_attribute.add_argument(
        "-c",
        "--skip_check_additivity",
        action="store_true",
        help="Skip check for additivity of attributions.",
    )

    # ISM SHUFFLE PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_ism_shuffle = subparsers.add_parser(
        "ism_shuffle",
        help="Calculate ISM shuffle scores for a given set of regions.",
        parents=[parser_parent],
    )
    parser_ism_shuffle.add_argument(
        "-d",
        "--n_dinucleotide_shuffles",
        type=int,
        default=5,
        help="Number of shuffles/mutations to perform for each position. "
        "Defaults to 5.",
    )
    parser_ism_shuffle.add_argument(
        "-s",
        "--mut_size",
        type=int,
        default=10,
        help="Size of mutations to use. That is, how large of a window to shuffle? "
        "Defaults to 10.",
    )
    parser_ism_shuffle.add_argument(
        "-e",
        "--edge_padding",
        type=int,
        default=50,
        help="Number of positions from edge that we'll skip mutating on.",
    )
    parser_ism_shuffle.add_argument(
        "-p",
        "--pseudocount",
        type=float,
        default=1e-6,
        help="Pseudocounts for calculations to avoid log or division by 0.",
    )

    # EPISTASIS PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_epistasis = subparsers.add_parser(
        "epistasis",
        help="Calculate Deep Feature Interaction Maps for a given set of regions.",
        parents=[parser_parent],
    )
    parser_epistasis.add_argument(
        "-a",
        "--attribution_type",
        type=str,
        default="quantity",
        choices={"quantity", "profile"},
        help="The type of attribution to calculate. REQUIRED if n_outputs == 2. "
        "if n_outputs == 1, this does nothing",
    )
    parser_epistasis.add_argument(
        "-s", "--start", type=int, default=250, help="Start position for DFIM."
    )
    parser_epistasis.add_argument(
        "-e", "--end", type=int, default=750, help="End position for DFIM."
    )
    parser_epistasis.add_argument(
        "-d",
        "--n_dinucleotide_shuffles",
        type=int,
        default=20,
        help="Maximum number of sequences to use as background. "
        "Default is 20 to ensure reasonably fast compute on medium-sized datasets. "
        "reducing to 5 for large datasets may be necessary.",
    )
    parser_epistasis.add_argument(
        "-c",
        "--skip_check_additivity",
        action="store_true",
        help="Disables check for additivity of shap results.",
    )

    # TSS PWM PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_tss = subparsers.add_parser(
        "tss_pwm",
        help="Calculate TSS PWMs for a given set of regions.",
        parents=[parser_parent],
    )
    parser_tss.add_argument(
        "-w", "--pwm_window", type=int, default=8, help="Size of PWM window."
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
            low_mem=True,
            silence=not args.verbose,
        )
        if args.n_outputs == 1:
            np.savez_compressed(args.output_fp, ensemble_predictions)
        else:
            np.savez_compressed(args.output_fp, *ensemble_predictions)

    # ATTRIBUTE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    elif args.cmd == "attribute":
        # Disable TF32 to reduce potential low precision issues
        tf.config.experimental.enable_tensor_float_32_execution(False)

        # Load data
        seqs_to_explain, twohot_background = attribute.load_seqs(
            fasta_fp=args.fasta_fp, n_subset=args.n_dinucleotide_shuffles
        )

        # Define contribution function
        if args.n_outputs == 1:
            contrib = attribute.scalar_contrib
        else:
            if args.attribution_type == "quantity":
                contrib = attribute.quantity_contrib
            elif args.attribution_type == "profile":
                contrib = attribute.profile_contrib

        # Create explainers
        if os.path.isdir(args.model_fp):
            model_names = list(glob.glob(os.path.join(args.model_fp, "*.h5")))
        else:
            model_names = [args.model_fp]
        explainers = attribute.create_explainers(
            model_fps=model_names,
            twohot_background=twohot_background,
            contrib=contrib,
            silence=len(model_names) == 1 or not args.verbose,
        )

        # Convert twohot to onehot and save
        if args.save_ohe is not None:
            np.savez_compressed(
                args.save_ohe, (seqs_to_explain / 2).astype(int).swapaxes(1, 2)
            )

        # Calculate attributions
        hyp_explanations = attribute.calculate_scores(
            explainers=explainers,
            seqs_to_explain=seqs_to_explain,
            batch_size=args.batch_size,
            silence=not args.verbose,
            check_additivity=not args.skip_check_additivity,
        )

        # Save attributions
        if args.save_hypothetical:
            np.savez_compressed(args.output_fp, hyp_explanations.swapaxes(1, 2))
        else:
            np.savez_compressed(
                args.output_fp,
                (hyp_explanations * (seqs_to_explain / 2)).swapaxes(1, 2),
            )

    # ISM SHUFFLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    elif args.cmd == "ism_shuffle":
        # Load models and sequences
        if os.path.isdir(args.model_fp):
            model = nn.construct_ensemble(args.model_fp, silence=not args.verbose)
        else:
            model = tf.keras.models.load_model(args.model_fp, compile=False)
        sequences = pyfastx.Fasta(args.fasta_fp)

        # Calculate ISM shuffle
        corr_scores, logfc_scores = ism_shuffle.ism_shuffle(
            model,
            sequences,
            mut_size=args.mut_size,
            n_shuffles=args.n_dinucleotide_shuffles,
            edge_padding=args.edge_padding,
            corr_pseudocount=args.pseudocount,
            logfc_pseudocount=args.pseudocount,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

        # Save
        np.savez_compressed(args.output_fp, corr_scores, logfc_scores)

    # EPISTASIS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    elif args.cmd == "epistasis":
        # Disable TF32 to reduce potential low precision issues
        tf.config.experimental.enable_tensor_float_32_execution(False)

        # Load sequences as strings
        seqs_to_explain, twohot_background = attribute.load_seqs(
            fasta_fp=args.fasta_fp,
            return_twohot_explains=False,
            n_subset=args.n_dinucleotide_shuffles,
        )

        # Define contribution function
        if args.n_outputs == 1:
            contrib = attribute.scalar_contrib
        else:
            if args.attribution_type == "quantity":
                contrib = attribute.quantity_contrib
            elif args.attribution_type == "profile":
                contrib = attribute.profile_contrib

        # Create explainers
        if os.path.isdir(args.model_fp):
            model_names = list(glob.glob(os.path.join(args.model_fp, "*.h5")))
        else:
            model_names = [args.model_fp]
        explainers = attribute.create_explainers(
            model_fps=model_names,
            twohot_background=twohot_background,
            contrib=contrib,
            silence=len(model_names) == 1 or not args.verbose,
        )

        # Calculate DFIM
        dfims = np.stack(
            [
                epistasis.dfim(
                    explainers=explainers,
                    major_seq=rec.seq,
                    start=args.start,
                    stop=args.end,
                    check_additivity=not args.skip_check_additivity,
                    silence=True,
                )
                for rec in tqdm.tqdm(
                    seqs_to_explain, desc="Calculating DFIM", disable=not args.verbose
                )
            ]
        )
        # Get extrema
        dfims = epistasis.extrema(dfims, axis=2).sum(axis=-1)

        # Save
        np.savez(args.output_fp, dfims)

    # TSS PWM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    elif args.cmd == "tss_pwm":
        tss = nn.compute_tss_pwm(args.model_fp, args.fasta_fp, window=args.pwm_window)
        file_ext = os.path.splitext(args.output_fp)[-1].strip(".")
        seqlogo.seqlogo(tss, format=file_ext, filename=args.output_fp, size="medium")

    # INVALID ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    cli()
