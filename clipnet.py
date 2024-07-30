"""
This file contains the CLIPNET class, which contains most of the main functions used to 
it, predict, and interpret the convolutional neural networks used in the CLIPNET project.
"""

import importlib
import json
import logging
import math
import os
import time
from pathlib import Path

import GPUtil
import numpy as np

import cgen
import utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "4"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf
import tqdm
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class CLIPNET:
    """
    Creates a CLIPNET instance.

    init args
    ------------
    n_gpus=1                - how many gpus to use.
    use_specific_gpu=0      - if n_gpus==1, allows choice of specific gpu.
    prefix='rnn_v10'        - prefix for nn_architecture file and the prefix the models
                              will be saved under while training.

    public functions
    ------------
    set_n_gpus              - Change the n_gpus to compute on.
    fit                     - Fit a model.
    predict_on_fasta        - Predicts on a given dataset.
    predict_ensemble        - Predicts on a given dataset using an ensemble of models.
    """

    def __init__(
        self,
        n_gpus=1,
        use_specific_gpu=0,
        prefix="rnn_v10",
    ):
        self.prefix = prefix
        self.n_gpus = n_gpus
        self.use_specific_gpu = use_specific_gpu
        self.nn = importlib.import_module(self.prefix)
        self.n_channels = 4
        self.__gpu_settings()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Core utilities
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __gpu_settings(self):
        if self.n_gpus == 0:
            print("Requested 0 GPUs. Turning off GPUs.")
            tf.config.set_visible_devices([], "GPU")
            self.strategy = tf.distribute.get_strategy()
        elif self.n_gpus >= 1:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if self.n_gpus == 1:
                gpus = tf.config.list_physical_devices("GPU")
                if self.use_specific_gpu is not None:
                    if self.use_specific_gpu >= len(gpus):
                        raise IndexError(
                            f"Requested GPU index {self.use_specific_gpu} does not exist."
                        )
                    gpu = gpus[self.use_specific_gpu]
                else:
                    gpu = gpus[GPUtil.getAvailable()[0]]
                print(f"Requested 1 GPU. Using GPU {gpu}.")
                tf.config.set_visible_devices(gpu, "GPU")
                self.strategy = tf.distribute.get_strategy()
            else:
                if self.n_gpus > len(gpus):
                    raise ValueError(
                        f"n_gpus ({self.n_gpus}) requested exceeds number of GPUs \
                        ({len(gpus)}) available."
                    )
                gpu_names = [gpu.name.split("physical_device:")[1] for gpu in gpus]
                self.strategy = tf.distribute.MirroredStrategy(
                    devices=gpu_names[: self.n_gpus]
                )
        else:
            raise ValueError("n_gpus must be a positive integer.")

    def set_n_gpus(self, n_gpus):
        """Reset number of GPUs."""
        self.n_gpus = n_gpus
        self.__gpu_settings()

    def __set_model_locations(self, resume_checkpoint):
        self.json_filepath = os.path.join(
            self.model_dir, f"{self.prefix}_architecture.json"
        )
        if resume_checkpoint is not None:
            self.model_filepath = os.path.join(self.model_dir, "clipnet_resume.hdf5")
        else:
            self.model_filepath = os.path.join(self.model_dir, "clipnet.hdf5")
        self.history_filepath = os.path.join(self.model_dir, "clipnet_history.json")

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
            sum(self.dataset_params["n_samples_per_train_fold"]) * 2 / batch_size
        )
        steps_per_val_epoch = math.floor(
            sum(self.dataset_params["n_samples_per_val_fold"]) * 2 / batch_size
        )
        return opt_hyperparameters, batch_size, steps_per_epoch, steps_per_val_epoch

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Model fitting and handling
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fit(
        self,
        model_dir,
        resume_checkpoint=None,
    ):
        """Fits a model based on specified arguments."""
        self.model_dir = model_dir
        with open(os.path.join(model_dir, "dataset_params.json"), "r") as handle:
            self.dataset_params = json.load(handle)
        self.__set_model_locations(resume_checkpoint)
        with self.strategy.scope():
            # adjust learning rate by n_gpus
            (
                opt_hyperparameters,
                batch_size,
                steps_per_epoch,
                steps_per_val_epoch,
            ) = self.__adjust_by_n_gpus()
            # load data
            train_args = {
                "seq_folds": self.dataset_params["train_seq"],
                "procap_folds": self.dataset_params["train_procap"],
                "steps_per_epoch": steps_per_epoch,
                "batch_size": batch_size,
                "pad": self.dataset_params["pad"],
            }
            val_args = {
                "seq_folds": self.dataset_params["val_seq"],
                "procap_folds": self.dataset_params["val_procap"],
                "steps_per_epoch": steps_per_val_epoch,
                "batch_size": batch_size,
                "pad": self.dataset_params["pad"],
            }
            train_gen = cgen.CGen(**train_args)
            val_gen = cgen.CGen(**val_args)
            # compile model
            if resume_checkpoint is not None:
                self.fit_model = tf.keras.models.load_model(
                    resume_checkpoint, compile=False
                )
                opt_hyperparameters, *_ = self.__adjust_by_n_gpus()
                self.fit_model.compile(
                    optimizer=self.nn.optimizer(**opt_hyperparameters),
                    loss=self.nn.loss,
                    loss_weights={"shape": 1, "sum": self.dataset_params["weight"]},
                    metrics=self.nn.metrics,
                )
                model = self.fit_model
            else:
                model = self.nn.construct_nn(
                    self.dataset_params["window_length"],
                    self.dataset_params["output_length"],
                )
                model.compile(
                    optimizer=self.nn.optimizer(**opt_hyperparameters),
                    loss=self.nn.loss,
                    loss_weights={"shape": 1, "sum": self.dataset_params["weight"]},
                    metrics=self.nn.metrics,
                )

            checkp = tf.keras.callbacks.ModelCheckpoint(
                self.model_filepath, verbose=0, save_best_only=True
            )
            early_stopping = tf.keras.callbacks.EarlyStopping(
                verbose=1, patience=self.nn.patience
            )
            training_time = TimeHistory()
            tqdm_callback = TqdmCallback(
                verbose=1, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            )
            csv_logger = CSVLogger(
                filename=os.path.join(self.model_dir, f"{self.prefix}.log"),
                separator=",",
                append=True,
            )

        # write model architecture
        with open(self.json_filepath, "w") as handle:
            handle.write(model.to_json())

        # fit model
        fit_model = model.fit(
            x=train_gen,
            validation_data=val_gen,
            epochs=self.nn.epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=0,
            callbacks=[
                checkp,
                early_stopping,
                training_time,
                tqdm_callback,
                csv_logger,
            ],
        )
        print("Compute times:")
        print(training_time.times)
        print("Saving model history ...")
        with open(self.history_filepath, "w") as f:
            json.dump(fit_model.history, f, indent=4)
        print(f"Successfully saved model history to {self.history_filepath}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Construct model ensemble.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def construct_ensemble(self, model_dir="./ensemble_models", silence=False):
        """
        Constructs an ensemble of models. Model ensembling is done by averaging the
        tracks and quantities of each model in the ensemble.
        """

        model_fps = list(Path(model_dir).glob("fold_*.h5"))
        models = [
            tf.keras.models.load_model(model_fp, compile=False)
            for model_fp in tqdm.tqdm(model_fps, desc="Loading models", disable=silence)
        ]
        for i in range(len(models)):
            models[i]._name = f"model_{i}"
        inputs = models[0].input
        tracks = [models[i](inputs)[0] for i in range(len(models))]
        quantities = [models[i](inputs)[1] for i in range(len(models))]
        outputs = [
            tf.keras.layers.Average()(tracks),
            tf.keras.layers.Average()(quantities),
        ]
        ensemble = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return ensemble

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Predict on fasta file.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def predict_on_fasta(
        self,
        model_fp,
        fasta_fp,
        reverse_complement=False,
        low_mem=True,
        desc="Predicting",
        silence=False,
    ):
        """
        Predicts on a fasta file, where each record is a 1000 5'-3' sequence.
        Returns [tracks, quantities].
        """
        if os.path.splitext(fasta_fp)[-1] == ".npz":
            sequence = np.load(fasta_fp)["arr_0"]
        elif os.path.splitext(fasta_fp)[-1] == ".npy":
            sequence = np.load(fasta_fp)
        else:
            sequence = utils.get_twohot_fasta_sequences(fasta_fp, silence=silence)
        X = utils.rc_twohot_het(sequence) if reverse_complement else sequence
        if os.path.isdir(model_fp):
            model = self.construct_ensemble(model_fp, silence=silence)
        else:
            model = tf.keras.models.load_model(model_fp, compile=False)
        if low_mem and self.nn.batch_size < X.shape[0]:
            # tensorflow has a memory leak issue in 2.?. We can fix this by using
            # model.__call__() on batches rather than the default predict.
            batch_size = self.nn.batch_size
            y_predict_handle = [
                model.predict(X[i : i + batch_size, :, :], verbose=0)
                for i in tqdm.tqdm(
                    range(0, X.shape[0], batch_size),
                    desc=f"Predicting in batches of {self.nn.batch_size}",
                    disable=silence,
                )
            ]
            y_predict = [
                np.concatenate([chunk[0] for chunk in y_predict_handle], axis=0),
                np.concatenate([chunk[1] for chunk in y_predict_handle], axis=0),
            ]
        else:
            y_predict = model.predict(X, batch_size=self.nn.batch_size, verbose=1)
        return y_predict

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute TSS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def compute_tss(self, model_dir, fasta_fp, window=8):
        """
        Computes the sequence at the predicted tss.
        """
        # compute predicted profile
        predicted_profile = self.predict_ensemble(
            model_dir=model_dir, fasta_fp=fasta_fp, low_mem=True
        )[0]
        # compute and return cpms
        predicted_tss_pos = np.argmax(
            predicted_profile[:, : int(predicted_profile.shape[1] / 2)], axis=1
        )
        return predicted_tss_pos

    def compute_tss_pwm(self, model_dir, fasta_fp, window=8):
        """
        Computes the sequence at the predicted tss.
        """
        import seqlogo

        tss = np.zeros((window, 4))
        # get sequences
        fwd_seq = utils.get_twohot_fasta_sequences(fasta_fp)
        rev_seq = utils.rc_twohot_het(fwd_seq)
        seq = np.concatenate((fwd_seq, rev_seq))
        # compute predicted profile
        fwd_profile = self.predict_ensemble(
            model_dir=model_dir, fasta_fp=fasta_fp, low_mem=True
        )[0]
        rev_profile = self.predict_ensemble(
            model_dir=model_dir,
            fasta_fp=fasta_fp,
            low_mem=True,
            reverse_complement=True,
        )[0]
        predicted_profile = np.concatenate((fwd_profile, rev_profile), axis=0)
        # compute and return cpms
        predicted_tss_pos = np.argmax(
            predicted_profile[:, : int(predicted_profile.shape[1] / 2)], axis=1
        )
        for i in range(predicted_tss_pos.shape[0]):
            if predicted_profile[i, predicted_tss_pos[i]] > 0:
                start = int(250 + predicted_tss_pos[i] - window / 2)
                stop = int(250 + predicted_tss_pos[i] + window / 2)
                subseq = seq[i, start:stop, :]
                tss += subseq
        return seqlogo.CompletePm(pfm=seqlogo.Pfm(tss))

    def get_activation_maps(
        self, model_fp, fasta_fp, predicted_tss_fp, layer=1, window=200
    ):
        """
        Computes activation maps for a given convolutional layer.
        """
        import joblib

        predicted_tss_pos = joblib.load(predicted_tss_fp)
        seq = utils.get_twohot_fasta_sequences(fasta_fp)

        with tf.device("/cpu:0"):
            model = tf.keras.models.load_model(model_fp, compile=False)
            conv_layer_outputs = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=model.get_layer("activation_%d" % layer).output,
            )
            activation = conv_layer_outputs.predict(seq)
            activation_around_tss = np.mean(
                np.stack(
                    [
                        activation[
                            i,
                            int((predicted_tss_pos[i] + 250 - window) / 2) : int(
                                (predicted_tss_pos[i] + 250 + window) / 2
                            ),
                        ]
                        for i in range(predicted_tss_pos.shape[0])
                    ]
                ),
                axis=0,
            )
        return activation_around_tss

    def get_filter_gc_content(
        self, model_fp, fasta_fp, layer=1, filter_width=15, n=5000
    ):
        """
        Computes activation maps for a given convolutional layer.
        """
        seq = utils.get_twohot_fasta_sequences(fasta_fp)
        with tf.device("/cpu:0"):
            model = tf.keras.models.load_model(model_fp, compile=False)
            conv_layer_outputs = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=model.get_layer("activation_%d" % layer).output,
            )
            activation = conv_layer_outputs.predict(seq)
            top_activations = activation.max(axis=1).argsort(axis=0)[-n:, :]
            gc_content = []
            for i in range(top_activations.shape[-1]):
                top_pos = activation[top_activations[:, i], :, i].argmax(axis=1)
                top_seqs = seq[top_activations[:, i], :, :]
                top_subseqs = [
                    top_seqs[
                        i,
                        top_pos[i] * (2**layer) : top_pos[i] * (2**layer)
                        + filter_width,
                        :,
                    ]
                    / 2
                    for i in range(top_seqs.shape[0])
                ]
                gc_content.append(
                    np.array(
                        [
                            np.sum(subseq[:, 1] + subseq[:, 2]) / filter_width
                            for subseq in top_subseqs
                        ]
                    ).mean()
                )
        return gc_content
