# CLIPNET

CLIPNET (Convolutionally Learned, Initiation-Predicting NETwork) is an ensembled convolutional neural network that predicts transcription initiation from DNA sequence at single nucleotide resolution. A preprint describing CLIPNET will be available shortly on bioRxiv. This repository contains code for working with CLIPNET, namely for generating predictions and feature interpretations. Code to reproduce the figures in our paper will be made available separately.

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

Pretrained CLIPNET models are available on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10408622). Download the models and unzip them into the `ensemble_models` directory:

```bash
for fold in {1..9};
do 
wget https://zenodo.org/records/10408623/files/fold_${fold}.h5 -O ensemble_models/fold_${fold}.h5;
done
```

## Usage

### Input data

CLIPNET was trained on a [population-scale PRO-cap dataset](http://dx.doi.org/10.1038/s41467-020-19829-z) derived from human lymphoblastoid cell lines, matched with individualized genome sequences (1kGP). CLIPNET accepts 1000 nt sequences as input and imputes PRO-cap coverage in the center 500 nt.

CLIPNET can either work on haploid reference sequences (e.g. hg38) or on individualized sequences (e.g. 1kGP). When constructing individualized sequences, we made two major simplifications: (1) We considered only SNPs and (2) we used unphased SNP genotypes.

### Command line interface

#### Predictions

To generate predictions, use the `predict.py` script. For example, to predict on the reference sequence:

```bash
python predict_ensemble.py \
    test.fa \
    test.h5
```
