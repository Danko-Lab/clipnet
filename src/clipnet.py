"""
This file contains the CLIPNET class, which contains most of the main functions used to 
it, predict, and interpret the residual neural networks used in the CLIPNET project.
"""

import glob
import json
import math
import os

import cgen
import GPUtil
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import time_history
import utils
from tqdm.keras import TqdmCallback
import seqlogo


class CLIPNET:
    """
    Creates a CLIPNET instance.

    init args
    ------------
    prefix='rnn_v9'         - prefix for nn_architecture file and the prefix the models
                              will be saved under.
    model_dir               - where to save/load models to/from.
    ='../../lcl_models/'
    dataset_params_fp       - a json file describing the dataset. Generate using
    ='./dataset_params.json'  calculate_dataset_params.py
    n_gpus=1                - how many gpus to train on.

    public functions
    ------------
    set_n_gpus              - Change the n_gpus to compute on.
    fit                     - Fit the model and save the outputs to self.model_dir.
    load_model              - Returns a fitted model from a specified directory.
    predict                 - Predicts on a given dataset.
    predict_metric          - Computes similarity metric between y_predict and y.
    compute_cpms            - Computes CPMs using seqlogo on a given activation layer.
    compute_activation_maps - Computes activation maps for a given layer.

    # FIX THESE
    """

    n_channels = 4 + 1  # DNA + DNase

    def __init__(
        self,
        prefix="rnn_v9",
        model_dir="../../lcl_models",
        dataset_params_fp="./dataset_params.json",
        n_gpus=1,
    ):
        import importlib

        self.prefix = prefix
        self.dataset_params_fp = dataset_params_fp
        with open(dataset_params_fp, "r") as handle:
            self.dataset_params = json.load(handle)
        self.n_gpus = n_gpus
        self.model_dir = model_dir
        self.nn = importlib.import_module(self.prefix)
        self.__gpu_settings()
        self.__set_model_locations()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Core utilities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __gpu_settings(self):
        assert self.n_gpus >= 0, "n_gpus must be an integer >= 0."
        if self.n_gpus == 0:
            print("Requested 0 GPUs. Turning off GPUs.")
            tf.config.set_visible_devices([], "GPU")
            self.strategy = tf.distribute.get_strategy()
        else:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if self.n_gpus == 1:
                gpus = tf.config.list_physical_devices("GPU")
                if len(gpus) >= 1:
                    tf.config.set_visible_devices(gpus[GPUtil.getAvailable()[0]], "GPU")
                self.strategy = tf.distribute.get_strategy()
            else:
                assert self.n_gpus <= len(
                    gpus
                ), f"n_gpus ({self.n_gpus}) requested exceeds number of GPUs \
                    ({len(gpus)}) available."
                gpu_names = [gpu.name.split("physical_device:")[1] for gpu in gpus]
                self.strategy = tf.distribute.MirroredStrategy(
                    devices=gpu_names[: self.n_gpus]
                )

    def set_n_gpus(self, n_gpus):
        """Reset number of GPUs."""
        self.n_gpus = n_gpus
        self.__gpu_settings()

    def __set_model_locations(self):
        self.json_filepath = os.path.join(
            self.model_dir, f"{self.prefix}_architecture.json"
        )
        self.model_filepath = os.path.join(
            self.model_dir, "%s_epoch{epoch:02d}.hdf5" % self.prefix
        )
        self.history_filepath = os.path.join(
            self.model_dir, f"{self.prefix}_history.joblib.gz"
        )
        self.model_summary_filepath = os.path.join(
            self.model_dir, f"{self.prefix}_model_summary.txt"
        )

    def __adjust_by_n_gpus(self):
        """This function adjusts parameters by the number of GPUs used in training."""
        if self.n_gpus == 0:
            n_gpus = 1
        else:
            n_gpus = self.n_gpus
        opt_hyperparameters = self.nn.opt_hyperparameters
        opt_hyperparameters["learning_rate"] = (
            n_gpus * self.nn.opt_hyperparameters["learning_rate"]
        )
        batch_size = n_gpus * self.nn.batch_size
        steps_per_epoch = math.floor(
            sum(self.dataset_params["n_samples_per_train_chunk"]) * 2 / batch_size
        )
        steps_per_val_epoch = math.floor(
            sum(self.dataset_params["n_samples_per_val_chunk"]) * 2 / batch_size
        )
        return opt_hyperparameters, batch_size, steps_per_epoch, steps_per_val_epoch

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Model fitting and handling
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fit(self, mixed_precision=False):
        """Fits a model based on specified arguments."""
        with self.strategy.scope():
            # adjust learning rate by n_gpus
            (
                opt_hyperparameters,
                batch_size,
                steps_per_epoch,
                steps_per_val_epoch,
            ) = self.__adjust_by_n_gpus()
            # load data
            train_args = [
                self.dataset_params["train_seq"],
                self.dataset_params["train_dnase"],
                self.dataset_params["train_procap"],
                steps_per_epoch,
                batch_size,
                self.dataset_params["pad"],
            ]
            val_args = [
                self.dataset_params["val_seq"],
                self.dataset_params["val_dnase"],
                self.dataset_params["val_procap"],
                steps_per_val_epoch,
                batch_size,
                self.dataset_params["pad"],
            ]
            train_gen = cgen.CGen(*train_args)
            val_gen = cgen.CGen(*val_args)
            # compile model
            model = self.nn.construct_nn(
                self.dataset_params["window_length"],
                self.dataset_params["output_length"],
            )
            model.compile(
                self.nn.optimizer(**opt_hyperparameters), self.nn.loss, self.nn.metrics
            )
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                self.model_filepath, verbose=0, save_best_only=True
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                verbose=1, patience=self.nn.patience
            )
            training_time = time_history.TimeHistory()
            tqdm_callback = TqdmCallback(
                verbose=1, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            )

        if mixed_precision is True:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        # write model summary and json architecture
        with open(self.model_summary_filepath, "w") as f:
            print(model.summary(), file=f)
        with open(self.json_filepath, "w") as handle:
            handle.write(model.to_json())

        # fit model
        fit_model = model.fit(
            x=train_gen,
            validation_data=val_gen,
            epochs=self.nn.epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=0,
            callbacks=[checkpoint, early_stopping, training_time, tqdm_callback],
        )
        print("Compute times:")
        print(training_time.times)
        print("Saving model history ...")
        joblib.dump(fit_model.history, self.history_filepath)
        print(f"Successfully saved model history to {self.history_filepath}")

    def load_model(self, model_filepath=None):
        """
        Loads model to self.fit_model If no filepath provided, attempts to load best
        model from self.model_dir.
        """
        with self.strategy.scope():
            if model_filepath is None:
                print(
                    f"No model file path provided. Searching for best model in \
                        {self.model_dir}"
                )
                pattern = os.path.join(self.model_dir, f"{self.prefix}_epoch*.hdf5")
                models = sorted(glob.glob(pattern))
                assert (
                    len(models) >= 1
                ), f"There are no models matching pattern {pattern}."
                model_filepath = models[-1]
            print(f"Attempting to load model from {model_filepath} ...")
            try:
                fit_model = tf.keras.models.load_model(model_filepath, compile=False)
            except OSError:
                raise OSError(
                    f"Cannot find model at {model_filepath}. Please make sure the \
                        model has been trained and the provided directory is correct."
                )
            print("Successfully loaded model.")
            # adjust learning rate by n_gpus
            opt_hyperparameters, *_ = self.__adjust_by_n_gpus()
            fit_model.compile(
                self.nn.optimizer(**opt_hyperparameters),
                self.nn.loss,
                [1, self.dataset_params["lambda"]],
                self.nn.metrics,
            )
        self.fit_model = fit_model

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Wrapper to allow class to easily load test data in dataset_params
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def load_test_data(
        self,
        test_seq=None,
        test_dnase=None,
        test_procap=None,
        reverse_complement=False,
        pad=None,
    ):
        """
        Wrapper for cgen.load_data that also handles loading defaults from
        self.dataset_params
        """
        names = (test_seq, test_dnase, test_procap)
        if all(fn is None for fn in names):
            keys = ["test_seq", "test_dnase", "test_procap"]
            assert all(
                k in self.dataset_params.keys() for k in keys
            ), "No test data file paths given or in dataset_params.json."
            print(
                f"No file names given. Defaulting to test data file names from \
                    {self.dataset_params_fp}."
            )
            names = (
                self.dataset_params["test_seq"],
                self.dataset_params["test_dnase"],
                self.dataset_params["test_procap"],
            )
        elif all(isinstance(fn, str) for fn in names):
            # All file names given, use these.
            pass
        else:
            raise TypeError(
                "Test data file names must all be None or a string. Received: %s %s %s"
                % names
            )
        if pad is None:
            pad = self.dataset_params["pad"]
        return cgen.load_data(*names, pad, reverse_complement)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Prediction wrappers
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def predict(
        self,
        test_seq=None,
        test_dnase=None,
        test_procap=None,
        data_Xy=None,
        reverse_complement=False,
    ):
        """
        Predicts on a full dataset containing sequence, dnase, and procap files. Returns
        a hash consisting of y_predict and y values.
        """
        if not hasattr(self, "fit_model"):
            self.load_model()
        if data_Xy is not None:
            X, y = data_Xy
        else:
            X, y = self.load_test_data(
                test_seq, test_dnase, test_procap, reverse_complement
            )
        print("Computing predictions ...")
        y_predict = self.fit_model.predict(X)
        print("Done predicting")
        return {"y_predict": y_predict, "y": y}

    def predict_metric(
        self,
        metric="pearson",
        test_seq=None,
        test_dnase=None,
        test_procap=None,
        reverse_complement=False,
    ):
        """Computes Pearson's correlation coefficient between predict_y and y."""
        assert metric in [
            "pearson",
            "cosine",
        ], "Metric provided not currently supported (pearson, cosine)."
        d = self.predict(
            test_seq=test_seq,
            test_dnase=test_dnase,
            test_procap=test_procap,
            reverse_complement=reverse_complement,
        )
        y_predict = pd.DataFrame(d["y_predict"][0])
        y = pd.DataFrame(d["y"])
        if metric == "pearson":
            return y_predict.corrwith(y, axis=1)
        elif metric == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity

            cosine = np.array(
                [
                    cosine_similarity(
                        y_predict[i, :].reshape(1, -1), y[i, :].reshape(1, -1)
                    )
                    for i in range(y_predict.shape[0])
                ]
            )
            return pd.DataFrame([c[0][0] for c in cosine])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Motif and motif distribution analyses
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_activations(self, activation_layer, X):
        """
        Returns activation matrix and subsampling factor (2 ** n(max_pooling_layers)).
        Memory intensive. Do not run on large X datasets (>100k windows).
        """
        if not hasattr(self, "fit_model"):
            self.load_model()
        assert activation_layer >= 0, "activation_layer >= 0 (use 0 for first layer)."
        if activation_layer == 0:
            layer_name = "activation"
        elif activation_layer >= 1:
            layer_name = f"activation_{activation_layer}"
        conv_layer_outputs = tf.keras.models.Model(
            inputs=self.fit_model.inputs,
            outputs=self.fit_model.get_layer(layer_name).output,
        )
        print("Computing activations ...")
        activations = conv_layer_outputs.predict(X)
        print("Successfully computed activations.")
        print(activations.shape)  # = (247658, 493, 128)
        layer_names = [layer.name for layer in conv_layer_outputs.layers]
        n_max_pooling = len([name for name in layer_names if "max_pooling" in name])
        return activations, 2**n_max_pooling

    def __compute_pfm(
        self, activations, subsample, X, filter_width, filter_num, n_samples=5000
    ):
        pfm = np.zeros((filter_width, self.n_channels))
        print(f"Starting {filter_num} out of {activations.shape[2]} positions.")
        # For a given filter, sort samples by descending activation
        sort_activation = np.argsort(np.max(activations[:, :, filter_num], axis=-1))[
            ::-1
        ]
        # Take top n_samples
        top_samples = sort_activation[:n_samples]
        # For each sample, filter, extract position in feature map with max activation
        max_position = np.argmax(activations[top_samples, :, filter_num], axis=1)
        # Retrieve subsequences for these positions
        for i in range(n_samples):
            if activations[top_samples[i], max_position[i], filter_num] > 0:
                subseq = X[
                    top_samples[i],
                    subsample * max_position[i] : subsample * max_position[i]
                    + filter_width,
                    :,
                ]
                if subseq.shape[0] < filter_width:
                    subseq = np.pad(
                        subseq, [(0, filter_width - subseq.shape[0]), (0, 0)]
                    )
                pfm += subseq
        pm = {
            "pfm": pfm[:, : self.n_channels - 1],
            "dnase": pfm[:, self.n_channels - 1],
        }
        return pm

    def compute_cpms(
        self,
        activation_layer,
        filter_width,
        test_seq=None,
        test_dnase=None,
        test_procap=None,
        n_samples=5000,
    ):
        """
        A wrapper for self.__compute_pfm. Returns a list of seqLogo.CompletePM objects.
        Must specify layer name, filter_width (compute separately), and test set.
        """
        # load data
        X_fwd, *_ = self.load_test_data(test_seq, test_dnase, test_procap)
        X_rev, *_ = self.load_test_data(
            test_seq, test_dnase, test_procap, reverse_complement=True
        )
        X = np.concatenate((X_fwd, X_rev))
        # compute activations and subsample
        activations, subsample = self.get_activations(activation_layer, X)
        pfms = []
        for filter_num in range(activations.shape[2]):
            pm = self.__compute_pfm(
                activations, subsample, X, filter_width, filter_num, n_samples
            )
            pfms.append(pm["pfm"])
        # compute and return cpms
        cpms = [seqlogo.CompletePm(pfm=seqlogo.Pfm(pfm)) for pfm in pfms]
        return cpms

    def get_activations_around_tss(
        self,
        activation_layer,
        test_seq=None,
        test_dnase=None,
        test_procap=None,
        map_window=200,
        side="both",
    ):
        """
        For each filter in activation_layer, extract the activations around the TSS.
        """
        assert side in ["both", "pl", "mn"], "side must be 'both', 'pl', or 'mn'."
        # Load data
        X_fwd, y_fwd = self.load_test_data(test_seq, test_dnase, test_procap)
        X_rev, y_rev = self.load_test_data(
            test_seq, test_dnase, test_procap, reverse_complement=True
        )
        # Concatenate reverse complement
        X = np.concatenate((X_fwd, X_rev))
        y = np.concatenate((y_fwd, y_rev))
        # predict and get max tsn
        predicted_profile = self.predict(data_Xy=(X, y))["y_predict"][0]
        tsn_pl = np.argmax(
            predicted_profile[:, : int(predicted_profile.shape[1] / 2)], axis=1
        )
        tsn_mn = np.argmax(
            predicted_profile[:, int(predicted_profile.shape[1] / 2) :], axis=1
        )
        # get activations
        activations, subsample = self.get_activations(activation_layer, X)
        # extract regions around max tss
        activation_pl = [
            activations[
                i,
                int(
                    (tsn_pl[i] + self.dataset_params["pad"] - map_window) / subsample
                ) : int(
                    (tsn_pl[i] + self.dataset_params["pad"] + map_window) / subsample
                ),
            ]
            for i in range(tsn_pl.shape[0])
        ]
        activation_mn = [
            activations[
                i,
                int(
                    (tsn_mn[i] + self.dataset_params["pad"] - map_window) / subsample
                ) : int(
                    (tsn_mn[i] + self.dataset_params["pad"] + map_window) / subsample
                ),
            ]
            for i in range(tsn_mn.shape[0])
        ]
        return np.stack(activation_pl), np.stack(activation_mn)

    def get_activations_around_tss_low_mem(
        self,
        activation_layer,
        test_seq=None,
        test_dnase=None,
        test_procap=None,
        map_window=200,
    ):
        """
        For each filter in activation_layer, extract the activations around the TSS.
        """
        # Load data
        X, y = self.load_test_data(test_seq, test_dnase, test_procap)
        # predict and get max tsn
        predicted_profile = self.predict(data_Xy=(X, y))["y_predict"][0]
        tsn_pl = np.argmax(
            predicted_profile[:, : int(predicted_profile.shape[1] / 2)], axis=1
        )
        # get activations
        activations, subsample = self.get_activations(activation_layer, X)
        # extract regions around max tss
        activation_pl = [
            activations[
                i,
                int(
                    (tsn_pl[i] + self.dataset_params["pad"] - map_window) / subsample
                ) : int(
                    (tsn_pl[i] + self.dataset_params["pad"] + map_window) / subsample
                ),
            ]
            for i in range(tsn_pl.shape[0])
        ]
        return np.stack(activation_pl)

    def compute_activation_maps(
        self,
        activation_layer,
        test_seq=None,
        test_dnase=None,
        test_procap=None,
        map_window=200,
        mode="mean",
    ):
        """
        For each filter in activation_layer, extract the activations around the TSS in
        each sample. Return averages or maxes across samples.
        """
        # Get activations
        activation_pl, activation_mn = self.get_activations_around_tss(
            activation_layer, test_seq, test_dnase, test_procap, map_window
        )
        # compute and return means
        assert mode in [
            "mean",
            "median",
            "max",
        ], "mode must be 'mean', 'median', or 'max'."
        if mode == "mean":
            activation_pl_maps = np.mean(activation_pl, axis=0)
            activation_mn_maps = np.mean(activation_mn, axis=0)
        elif mode == "max":
            activation_pl_maps = np.max(activation_pl, axis=0)
            activation_mn_maps = np.max(activation_mn, axis=0)
        else:
            activation_pl_maps = np.median(activation_pl, axis=0)
            activation_mn_maps = np.median(activation_mn, axis=0)
        return {"pl": activation_pl_maps, "mn": activation_mn_maps}

    def get_max_sequences(
        self,
        activation_layer,
        test_seq=None,
        test_dnase=None,
        test_procap=None,
        n_samples=5000,
    ):
        """
        For each filter, get the sequence windows (and positions in each window) with
        the maximum activation.

        NOTE: THIS IS FOR AN ANALYSIS THAT DID NOT WORK. CONSIDER DELETING.
        """
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        from utils import RevOneHotDNA

        X, _ = self.load_test_data(test_seq, test_dnase, test_procap)
        activations, _ = self.get_activations(activation_layer, X)
        records = {}
        dnase = {}
        for filter_num in range(activations.shape[2]):
            # For a given filter, sort samples by descending activation
            sort_activation = np.argsort(
                np.max(activations[:, :, filter_num], axis=-1)
            )[::-1]
            # Take top n_samples
            top_samples = sort_activation[:n_samples]
            # For each sample, filter, get position in feature map with max activation
            max_position = np.argmax(activations[top_samples, :, filter_num], axis=1)
            records[filter_num] = (
                SeqRecord(
                    Seq(RevOneHotDNA(X[top_samples[i], :, :4]).seq),
                    id=f"seq={i};max_pos={max_position[i]}",
                )
                for i in range(n_samples)
            )
            dnase[filter_num] = X[top_samples, :, 4]
        return records, dnase

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Predict on fasta file (for mutagenesis and contribution score analyses).
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def predict_on_fasta(
        self, fasta_fp, dnase_fp, procap_fp=None, reverse_complement=False
    ):
        """
        Predicts on a fasta and dnase.csv file, where each record is a 1000 5'-3'
        sequence and dnase data. Returns y_predict.
        """

        if not hasattr(self, "fit_model"):
            self.load_model()
        sequence = utils.get_onehot_fasta_sequences(fasta_fp)
        dnase = pd.read_csv(dnase_fp, index_col=0, header=None)
        if reverse_complement:
            X = np.dstack((utils.rc_onehot_het(sequence), dnase[dnase.columns[::-1]]))
        else:
            X = np.dstack((sequence, dnase))
        print("Computing predictions ...")
        y_predict = self.fit_model.predict(X)
        print("Done predicting")
        if procap_fp is None:
            return y_predict
        else:
            y = pd.read_csv(procap_fp, index_col=0, header=None)
            return {"y_predict": y_predict, "y": y}

    def compute_tss(self, fasta_fp, dnase_fp, window=8):
        """
        Computes the sequence at the predicted tss.
        """
        import seqlogo
        import utils

        tss = np.zeros((window, 4))
        pad = int(window / 2)
        # get sequences
        fwd_seq = utils.get_onehot_fasta_sequences(fasta_fp)
        rev_seq = utils.rc_onehot_het(fwd_seq)
        seq = np.concatenate((fwd_seq, rev_seq))
        # compute predicted profile
        fwd_profile = self.predict_on_fasta(fasta_fp, dnase_fp)[0]
        rev_profile = self.predict_on_fasta(
            fasta_fp, dnase_fp, reverse_complement=True
        )[0]
        predicted_profile = np.concatenate((fwd_profile, rev_profile))
        # compute and return cpms
        predicted_tss_pos = np.argmax(
            predicted_profile[:, : int(predicted_profile.shape[1] / 2)], axis=1
        )
        for i in range(predicted_tss_pos.shape[0]):
            if predicted_profile[i, predicted_tss_pos[i]] > 0:
                start = self.dataset_params["pad"] + predicted_tss_pos[i] - pad
                stop = self.dataset_params["pad"] + predicted_tss_pos[i] + pad
                subseq = seq[i, start:stop, :]
                tss += subseq
        return seqlogo.CompletePm(pfm=seqlogo.Pfm(tss))
