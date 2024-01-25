# CLIPNET

CLIPNET (Convolutionally Learned, Initiation-Predicting NETwork) is an ensembled convolutional neural network that predicts transcription initiation from DNA sequence at single nucleotide resolution. A preprint describing CLIPNET will be available shortly on bioRxiv. This repository contains code for working with CLIPNET, namely for generating predictions and feature interpretations and performing *in silico* mutagenesis scans. Code to reproduce the figures in our paper will be made available separately.

## Installation

To install CLIPNET, first clone this repository:

```bash
git clone https://github.com/Danko-Lab/clipnet.git
cd clipnet
```

Then, install prerequisites using pip. We recommend creating an isolated environment for working with CLIPNET. For example, using conda/mamba:

```bash
conda create -n clipnet python=3.9 # or mamba
conda activate clipnet
pip install -r tf_requirements.txt
```

We had issues with conflicting package requirements when installing DeepSHAP, so we recommend installing it separately:

```bash
conda create -n shap python=3.9 # or mamba
conda activate shap
pip install -r shap_requirements.txt
```

## Download models

Pretrained CLIPNET models are available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10408622). Download the models and into the `ensemble_models` directory:

```bash
for fold in {1..9};
do wget https://zenodo.org/records/10408623/files/fold_${fold}.h5 -P ensemble_models/;
done
```

## Usage

### Input data

CLIPNET was trained on a [population-scale PRO-cap dataset](http://dx.doi.org/10.1038/s41467-020-19829-z) derived from human lymphoblastoid cell lines, matched with individualized genome sequences (1kGP). CLIPNET accepts 1000 bp sequences as input and imputes PRO-cap coverage (RPM) in the center 500 bp.

CLIPNET can either work on haploid reference sequences (e.g. hg38) or on individualized sequences (e.g. 1kGP). When constructing individualized sequences, we made two major simplifications: (1) We considered only SNPs and (2) we used unphased SNP genotypes.

We encode sequences using a "two-hot" encoding. That is, we encoded each individual nucleotide at a given position using a one-hot encoding scheme, then represented the unphased diploid sequence as the sum of the two one-hot encoded nucleotides at each position. The sequence AYCR, for example, would be encoded as [[2, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 0], [1, 0, 1, 0]].

### Command line interface

#### Predictions

To generate predictions using the ensembled model, use the `predict_ensemble.py` script (the `predict_individual_models.py` script can be used to generate predictions with individual model folds). This script takes a fasta file containing 1000 bp records and outputs an hdf5 file containing the predictions for each record. For example:

```bash
# conda activate tf
python predict_ensemble.py data/test.fa data/test.h5 --gpu
```

To input individualized sequences, heterozygous positions should be represented using the IUPAC ambiguity codes R (A/G), Y (C/T), S (C/G), W (A/T), K (G/T), M (A/C).

The output hdf5 file will contain two datasets: "track" and "quantity". The track output of the model is a length 1000 vector (500 plus strand concatenated with 500 minus strand) representing the predicted base-resolution profile/shape of initiation. The quantity output represents the total PRO-cap quantity on both strands.

We note that the track node was not optimized for quantity prediction, and that the sum of the track node is not well correlated with the quantity prediction, and not a good predictor of the total quantity of initiation. We therefore recommend rescaling the track predictions to sum to the quantity prediction. For example:

```python
import h5py
import numpy as np

with h5py.File("data/test.h5", "r") as f:
    profile = f["track"][:]
    quantity = f["quantity"][:]
    profile_scaled = profile * quantity[:, None] / np.sum(profile, axis=1)[:, None]
```

#### Feature interpretations

CLIPNET uses DeepSHAP to generate feature interpretations. To generate feature interpretations, use the `calculate_deepshap.py` script. This script takes a fasta file containing 1000 bp records and outputs two npz files containing: (1) feature interpretations for each record and (2) onehot-encoded sequence. These files are required as input for [`tfmodisco-lite`](https://github.com/jmschrei/tfmodisco-lite/tree/main). Since calculating these interpretations can be quite slow when run on large datasets, this script is set up to calculate for a single model fold. This enables easy multi-GPU calculation of DeepSHAP scores, since you can just run one script on each GPU.

This script supports two modes: "profile" and "quantity". The "profile" mode calculates interpretations for the profile node of the model (using the profile metric proposed in BPNet), while the "quantity" mode calculates interpretations for the quantity node of the model. For example:

```bash
# conda activate shap
python calculate_deepshap.py \
    ensemble_models/fold_1.h5 \
    data/test.fa \
    data/test_deepshap.npz \
    data/test_onehot.npz \
    --mode quantity \
    --gpu
```

Note that CLIPNET generally accepts two-hot encoded sequences as input, with the array being structured as (# sequences, 1000, 4). However, feature interpretations are much easier to do with just a haploid/fully homozygous genome, so we recommend just doing interpretations on the reference genome sequence. `tfmodisco-lite` also expects contribution scores and sequence arrays to be length last, i.e., (# sequences, 4, 1000). To accomodate these, `calculate_deepshap.py` will automatically convert the input sequence array to length last and onehot encoded, and will also write the output contribution scores as length last.

Also note that these are actual contribution scores, as opposed to hypothetical contribution scores. Specifically, non-reference nucleotides are set to zero.

The outputs of this model can be used as input to `tfmodisco-lite` to generate motif logos and motif tracks.

#### Genomic *in silico* mutagenesis scans

To generate genomic *in silico* mutagenesis scans, use the `calculate_ism_shuffle.py` script. This script takes a fasta file containing 1000 bp records and outputs an hdf5 file containing the predictions for each record. For example:

```bash
# conda activate tf
python calculate_ism_shuffle.py data/test.fa data/test_ism.npz --gpu
```

#### API usage

CLIPNET models can be directly loaded as follows. Individual models can simply be loaded using `tensorflow`:

```python
import tensorflow as tf

nn = tf.keras.models.load_model("ensemble_models/fold_1.h5", compile=False)
```

The model ensemble is constructed by averaging the track and quantity outputs of each model fold. To make this easy, we've provided a simple API in the `clipnet.CLIPNET` class for doing this. Moreover, to make reading fasta files into the correct format easier, we've provided the helpder function `utils.twohot_fasta`. For example:

```python
import sys
sys.path.append(PATH_TO_THIS_DIRECTORY)
import clipnet
import utils

nn = clipnet.CLIPNET(n_gpus=1)
ensemble = nn.construct_ensemble()
seqs = utils.twohot_fasta("data/test.fa")

predictions = ensemble.predict(seqs)
```
