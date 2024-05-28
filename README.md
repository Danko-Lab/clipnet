# CLIPNET

CLIPNET (Convolutionally Learned, Initiation-Predicting NETwork) is an ensembled convolutional neural network that predicts transcription initiation from DNA sequence at single nucleotide resolution. We describe CLIPNET in our [preprint](https://www.biorxiv.org/content/10.1101/2024.03.13.583868) on bioRxiv. This repository contains code for working with CLIPNET, namely for generating predictions and feature interpretations and performing *in silico* mutagenesis scans. To reproduce the figures in our paper, please see the [clipnet_paper GitHub repo](https://github.com/Danko-Lab/clipnet_paper/).

## Installation

To install CLIPNET, first clone this repository:

```bash
git clone https://github.com/Danko-Lab/clipnet.git
cd clipnet
```

Then, install dependencies using pip. We recommend creating an isolated environment for working with CLIPNET. For example, with conda/mamba:

```bash
mamba create -n clipnet -c conda-forge gcc~=12.1 python=3.9
mamba activate clipnet
pip install -r requirements.txt # requirements_cpu.txt if no GPU
```

You may need to configure your CUDA/cudatoolkit/cudnn paths to get GPU support working. See the [tensorflow documentation](https://www.tensorflow.org/install/gpu) for more information.

## Download models

Pretrained CLIPNET models are available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10408622). Download the models into the `ensemble_models` directory:

```bash
for fold in {1..9};
do wget https://zenodo.org/records/10408623/files/fold_${fold}.h5 -P ensemble_models/;
done
```

Alternatively, they can be accessed via [HuggingFace](https://huggingface.co/adamyhe/clipnet).

## Usage

### Input data

CLIPNET was trained on a [population-scale PRO-cap dataset](http://dx.doi.org/10.1038/s41467-020-19829-z) derived from human lymphoblastoid cell lines, matched with individualized genome sequences (1kGP). CLIPNET accepts 1000 bp sequences as input and imputes PRO-cap coverage (RPM) in the center 500 bp.

CLIPNET can either work on haploid reference sequences (e.g. hg38) or on individualized sequences (e.g. 1kGP). When constructing individualized sequences, we made two major simplifications: (1) We considered only SNPs and (2) we used unphased SNP genotypes.

We encode sequences using a "two-hot" encoding. That is, we encoded each individual nucleotide at a given position using a one-hot encoding scheme, then represented the unphased diploid sequence as the sum of the two one-hot encoded nucleotides at each position. The sequence "AYCR", for example, would be encoded as: `[[2, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 0], [1, 0, 1, 0]]`.

### Command line interface

#### Predictions

To generate predictions using the ensembled model, use the `predict_ensemble.py` script (the `predict_individual_model.py` script can be used to generate predictions with individual model folds). This script takes a fasta file containing 1000 bp records and outputs an hdf5 file containing the predictions for each record. For example:

```bash
python predict_ensemble.py data/test.fa data/test_predictions.h5 --gpu 0
# Use the --gpu flag to select which GPU to run on
```

To input individualized sequences, heterozygous positions should be represented using the IUPAC ambiguity codes R (A/G), Y (C/T), S (C/G), W (A/T), K (G/T), M (A/C).

The output hdf5 file will contain two datasets: "track" and "quantity". The track output of the model is a length 1000 vector (500 plus strand concatenated with 500 minus strand) representing the predicted base-resolution profile/shape of initiation. The quantity output represents the total PRO-cap quantity on both strands.

We note that the track node was not optimized for quantity prediction. As a result, the sum of the track node is not well correlated with the quantity prediction and not a good predictor of the total quantity of initiation. We therefore recommend rescaling the track predictions to sum to the quantity prediction. For example:

```python
import h5py
import numpy as np

with h5py.File("data/test_predictions.h5", "r") as f:
    profile = f["track"][:]
    quantity = f["quantity"][:]
    profile_scaled = (profile / np.sum(profile, axis=1)[:, None]) * quantity
```

#### Feature interpretations

CLIPNET uses DeepSHAP to generate feature interpretations. To generate feature interpretations, use the `calculate_deepshap.py` script. This script takes a fasta file containing 1000 bp records and outputs two npz files containing: (1) feature interpretations for each record and (2) onehot-encoded sequence. It supports two modes that can be set with `--mode`: "profile" and "quantity". The "profile" mode calculates interpretations for the profile node of the model (using the profile metric proposed in BPNet), while the "quantity" mode calculates interpretations for the quantity node of the model.

```bash
python calculate_deepshap.py \
    data/test.fa \
    data/test_deepshap_quantity.npz \
    data/test_onehot.npz \
    --mode quantity \
    --gpu 0

python calculate_deepshap.py \
    data/test.fa \
    data/test_deepshap_profile.npz \
    data/test_onehot.npz \
    --mode profile \
    --gpu 0
```

Note that CLIPNET generally accepts two-hot encoded sequences as input, with the array being structured as (# sequences, 1000, 4). However, feature interpretations are much easier to do with just a haploid/fully homozygous genome, so we recommend just doing interpretations on the reference genome sequence. tfmodisco-lite also expects contribution scores and sequence arrays to be length last, i.e., (# sequences, 4, 1000), with the sequence array being one-hot. To accomodate these, `calculate_deepshap.py` will automatically convert the input sequence array to length last and onehot encoded, and will also write the output contribution scores as length last. Also note that these are actual contribution scores, as opposed to hypothetical contribution scores. Specifically, non-reference nucleotides are set to zero. The outputs of this model can be used as inputs to tfmodisco-lite to generate motif logos and motif tracks.

Both DeepSHAP and tfmodisco-lite computations are quite slow when performed on a large number of sequences, so we (a) recommend running DeepSHAP on a GPU using the `--gpu` flag and (b) if you have access to many GPUs, calculating DeepSHAP scores for the model folds in parallel using the `--model_fp` flag, then averaging them. We also provide precomputed DeepSHAP scores and TF-MoDISco results for a genome-wide set of PRO-cap peaks called in the LCL dataset (https://zenodo.org/records/10597358).

#### Genomic *in silico* mutagenesis scans

To generate genomic *in silico* mutagenesis scans, use the `calculate_ism_shuffle.py` script. This script takes a fasta file containing 1000 bp records and outputs an npz file containing the ISM shuffle results ("corr_ism_shuffle" and "log_quantity_ism_shuffle") for each record. For example:

```bash
python calculate_ism_shuffle.py data/test.fa data/test_ism.npz --gpu 0
```

### API usage

CLIPNET models can be directly loaded as follows. Individual models can simply be loaded using tensorflow:

```python
import tensorflow as tf

nn = tf.keras.models.load_model("ensemble_models/fold_1.h5", compile=False)
```

The model ensemble is constructed by averaging track and quantity outputs across all 9 model folds. To make this easy, we've provided a simple API in the `clipnet.CLIPNET` class for doing this. Moreover, to make reading fasta files into the correct format easier, we've provided the helper function `utils.twohot_fasta`. For example:

```python
import sys
sys.path.append(PATH_TO_THIS_DIRECTORY)
import clipnet
import utils

nn = clipnet.CLIPNET(n_gpus=0) # by default, this will be 1 and will use CUDA
ensemble = nn.construct_ensemble()
seqs = utils.twohot_fasta("data/test.fa")

predictions = ensemble.predict(seqs)
```
