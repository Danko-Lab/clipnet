# CLIPNET

CLIPNET (Convolutionally Learned, Initiation-Predicting NETwork) is an ensembled convolutional neural network that predicts transcription initiation from DNA sequence at single nucleotide resolution. We describe CLIPNET in our [preprint](https://www.biorxiv.org/content/10.1101/2024.03.13.583868) on bioRxiv. This repository contains code for working with CLIPNET, namely for generating predictions and feature interpretations and performing *in silico* mutagenesis scans. To reproduce the figures in our paper, please see the [clipnet_paper GitHub repo](https://github.com/Danko-Lab/clipnet_paper/).

## PyTorch reimplementation and port

Code to port the TensorFlow models to PyTorch is available as part of [PersonalBPNet](https://github.com/adamyhe/PersonalBPNet/), which also includes a from-scratch reimplementation of CLIPNET in PyTorch with a context length of 2114 bp.

## CODE REFACTORING NOTICE

I have significantly altered the structure of this code base since its original release with the preprint. The new CLIPNET package should be significantly easier to use (`pip` installable, with clearer CLI and API). To access the code as it was prior to this refactoring, please check out the (unmaintained) [`deprecated`](https://github.com/Danko-Lab/clipnet/tree/deprecated) branch.

## Installation

To install CLIPNET, we recommend creating an isolated environment. For example, with conda/mamba:

```bash
mamba create -n clipnet -c conda-forge python~=3.9
mamba activate clipnet
```

Then clone this repo and install with `pip`:

```bash
git clone https://github.com/Danko-Lab/clipnet.git
cd clipnet/
pip install -e .
```

You may need to configure your CUDA/cudatoolkit/cudnn paths to get GPU support working. See the [tensorflow documentation](https://www.tensorflow.org/install/gpu) for more information.

## Download models

Pretrained CLIPNET models are available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10408622).

```bash
mkdir -p clipnet_models/
for fold in {1..9};
do wget https://zenodo.org/records/10408623/files/fold_${fold}.h5 -P clipnet_models/;
done
```

Alternatively, they can be accessed via [HuggingFace](https://huggingface.co/adamyhe/clipnet).

## Usage

### Input data

CLIPNET was trained on a [population-scale PRO-cap dataset](http://dx.doi.org/10.1038/s41467-020-19829-z) derived from human lymphoblastoid cell lines, matched with individualized genome sequences (1kGP). CLIPNET accepts 1000 bp sequences as input and imputes PRO-cap coverage (RPM) in the center 500 bp.

CLIPNET can either work on haploid reference sequences (e.g. hg38) or on individualized sequences (e.g. 1kGP). When constructing individualized sequences, we made two major simplifications: (1) We considered only SNPs and (2) we used unphased SNP genotypes.

We encode sequences using a "two-hot" encoding. That is, we encoded each individual nucleotide at a given position using a one-hot encoding scheme, then represented the unphased diploid sequence as the sum of the two one-hot encoded nucleotides at each position. The sequence "AYCR" (= A(C/T)C(A/G)), for example, would be encoded as: `[[2, 0, 0, 0], [0, 1, 0, 1], [0, 2, 0, 0], [1, 0, 1, 0]]`.

### Command line interface

CLIPNET can be accessed via a CLI:

```bash
clipnet -h
```

#### Predictions

The `predict` command can be used to generate predictions:

```bash
clipnet predict -f data/test.fa -o data/test_predictions.npz -m clipnet_models/ -v
```

The `-m` flag should be used to specify either a path to the directory containing the CLIPNET models (in which case the averaged predictions across all model replicates will be returned) or a specific model path (in which case only the predictions of that model will be returned).

To input individualized sequences, heterozygous positions should be represented using the IUPAC ambiguity codes R (A/G), Y (C/T), S (C/G), W (A/T), K (G/T), M (A/C).

The output npz file will contain two arrays. The first output (`"arr_0"`, "profile") is a length 1000 vector (500 plus strand concatenated with 500 minus strand) representing the predicted base-resolution profile/shape of initiation. The second output (`"arr_0"`, "quantity") represents the total PRO-cap quantity on both strands.

To generate actual predicted tracks, the profile prediction should be rescaled by the quantity prediction. For example:

```python
import numpy as np

f = np.load("data/test_predictions.npz") 
profile = f["arr_0"]
quantity = f["arr_1"]
profile_scaled = (profile / np.sum(profile, axis=1)[:, None]) * quantity
```

#### Attributions

CLIPNET uses [DeepSHAP](https://shap.readthedocs.io/en/latest/generated/shap.DeepExplainer.html) to generate attributions. To generate DeepSHAP scores, use the `attribute` command. This script takes a fasta file containing 1000 bp records and outputs DeepSHAP attributions and optionally one-hot encoded sequences. Please note that both attribution and ohe are saved as length last for compatibility with [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite/).

Two different attribution modes that can be set with `-a/--attribution_type`: `profile` and `quantity`. The `profile` mode calculates interpretations for the profile node of the model (using the profile metric proposed in BPNet), while the `quantity` mode calculates interpretations for the quantity node of the model.

```bash
clipnet attribute \
    -f data/test.fa.gz \
    -o data/test_quantity_shap.npz \
    -m clipnet_models/ \
    -a quantity \
    -v

# -c maybe needed to avoid precision errors.
```

Note that while CLIPNET accepts two-hot encoded sequences to accomodate heterozygous positions, attributions are much more interpretable when using a haploid/fully homozygous genome, so we recommend avoiding heterozygous positions for attributions. Also note that these are actual contribution scores, as opposed to hypothetical contribution scores. Specifically, non-reference nucleotides are set to zero. To return attribution scores for all nucleotides, use the `-y` flag.

#### Discovering epistatic motifs

`clipnet` supports epistasis analyses using [Deep Feature Interaction Maps (DFIM)](https://github.com/kundajelab/dfim). Please note that this is a custom reimplementation of DFIM using DeepSHAP as the attribution backend, as the original DFIM package is unmaintained and difficult to install. DFIM scores can be calculated for a given fasta file using the `epistasis` command:

```bash
clipnet epistasis \
    -f data/test.fa \
    -o data/test_dfim_profile.npz \
    -m clipnet_models/ \
    -s 250 -e 750 \
    -a profile \
    -v
```

Please note DFIM scores don't properly account for things like global epistasis/nonlinearity, which can cause misleading interpretations. For a more robust (but time-consuming) method for estimating interaction effects, see [SQUID](https://github.com/evanseitz/squid-nn).

#### Genomic *in silico* mutagenesis scans

To generate genomic *in silico* mutagenesis scans, use the `ism_shuffle` script. This script takes a fasta file containing 1000 bp records and outputs an npz file containing the ISM shuffle results (`corr_ism_shuffle` and `logfc_ism_shuffle`) for each record. For example:

```bash
clipnet ism_shuffle -f data/test.fa -o data/test_ism.npz -m clipnet_models/ -v
```

### API usage

CLIPNET models can be directly loaded as follows. Individual models can simply be loaded using tensorflow:

```python
import tensorflow as tf

nn = tf.keras.models.load_model("clipnet_models/fold_1.h5", compile=False)
```

The model ensemble is constructed by averaging track and quantity outputs across all 9 model folds. To make this easy, we've provided a simple API in the `clipnet.clipnet.CLIPNET` class for doing this. Moreover, to make reading fasta files into the correct format easier, we've provided the helper function `clipnet.utils.get_twohot_fasta_sequences`. For example:

```python
import sys
from clipnet.clipnet import CLIPNET
from clipnet.utils import get_twohot_fasta_sequences

nn = CLIPNET()
ensemble = nn.construct_ensemble("clipnet_models/")
seqs = get_twohot_fasta_sequences("data/test.fa")

predictions = ensemble.predict(seqs)
```
