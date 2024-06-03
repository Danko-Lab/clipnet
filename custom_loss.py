"""This implements a number of custom loss/metric functions for use in CLIPNET."""

import tensorflow as tf
import tensorflow_probability as tfp


def pearson_loss(y_true, y_pred):
    """Deprecated. Use correlation_loss."""
    true_residual = y_true - tf.math.mean(y_true)
    pred_residual = y_pred - tf.math.mean(y_pred)
    num = tf.math.sum(tf.math.multiply(true_residual, pred_residual))
    den = tf.math.sqrt(
        tf.math.multiply(
            tf.math.sum(tf.math.square(true_residual)),
            tf.math.sum(tf.math.square(pred_residual)),
        )
    )
    r = num / den
    return -r  # makes function decreasing and non-zero


def corr(x, y, pseudocount=1e-6):
    """
    Computes Pearson's r between x and y. Pseudocount ensures non-zero denominator.
    """
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    num = tf.math.reduce_mean(tf.multiply(xm, ym))
    den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym) + pseudocount
    r = tf.math.maximum(tf.math.minimum(num / den, 1), -1)
    return r


def corr_loss(x, y, pseudocount=1e-6):
    """Computes -correlation(x, y)."""
    return -corr(x, y, pseudocount)


def squared_log_sum_error(x, y, pseudocount=1e-6):
    """
    Computes the squared difference between log sums of vectors. Pseudocount ensures
    non-zero log inputs.
    """
    log_sum_x = tf.math.log(tf.math.reduce_sum(x) + pseudocount)
    log_sum_y = tf.math.log(tf.math.reduce_sum(y) + pseudocount)
    return (log_sum_x - log_sum_y) ** 2


def cosine_slse(x, y, slse_scale=8e-3, pseudocount=1e-6):
    """Computes cosine loss + scale * slse."""
    cosine_loss = tf.keras.losses.CosineSimilarity()
    return cosine_loss(x, y).numpy() + slse_scale * squared_log_sum_error(
        x, y, pseudocount
    )


def sum_error(x, y):
    return tf.math.reduce_sum(x) - tf.math.reduce_sum(y)


def sum_true(x, y):
    return tf.math.reduce_sum(x)


def sum_pred(x, y):
    return tf.math.reduce_sum(y)


def jaccard_distance(y_true, y_pred, smooth=100):
    """Calculates mean of Jaccard distance as a loss function"""
    y = tf.cast(y_true, tf.float32)
    intersection = tf.math.reduce_sum(y * y_pred)
    sum_ = tf.math.reduce_sum(y + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = (1 - jac) * smooth
    return jd  # tf.reduce_mean(jd)


def multinomial_nll(true_counts, logits):
    """
    Compute the multinomial negative log-likelihood along the sequence (axis=1)
    and sum the values across all channels

    Adapted from Avsec et al. (2021) Nature Genetics. https://doi.org/10.1038/s41588-021-00782-6
    Args:
      true_counts: observed count values (batch, seqlen, channels)
      logits: predicted logit values (batch, seqlen, channels)
    """
    # round sum to nearest int
    counts_per_example = tf.math.round(tf.reduce_sum(true_counts, axis=-1))
    # compute distribution
    dist = tf.compat.v1.distributions.Multinomial(
        total_count=counts_per_example, logits=logits
    )
    # return negative log probabilities
    return -tf.reduce_sum(dist.log_prob(true_counts))


class PearsonCorrelation(tf.keras.metrics.Mean):
    def __init__(self, post_proc_fn=None, **kwargs):
        """Pearson Correlation Coefficient. Taken from:
        https://gist.github.com/mhorlacher/8599d1204a48caa172e9ecfc0c73b989

        Args:
            post_proc_fn (function, optional): Post-processing function for predicted values. Defaults to None.
        """

        super().__init__(name="pearson_correlation", **kwargs)

        self.post_proc_fn = post_proc_fn
        if self.post_proc_fn is None:
            self.post_proc_fn = lambda y, y_pred: (y, y_pred)

    def _compute_correlation(self, y, y_pred):
        corr = tfp.stats.correlation(y, y_pred, sample_axis=1, event_axis=-1)
        return corr

    def _nan_to_zero(self, x):
        is_not_nan = tf.math.logical_not(tf.math.is_nan(x))
        is_not_nan = tf.cast(is_not_nan, tf.float32)

        return tf.math.multiply_no_nan(x, is_not_nan)

    def update_state(self, y, y_pred, **kwargs):
        y, y_pred = self.post_proc_fn(y, y_pred)

        corr = self._compute_correlation(y, y_pred)
        corr = tf.squeeze(corr)

        # remove any nan's that could have been created, e.g. if y or y_pred is a 0-vector
        corr = self._nan_to_zero(corr)

        # assert that there are no inf's or nan's
        tf.debugging.assert_all_finite(corr, f"expected finite tensor, got {corr}")

        super().update_state(corr, **kwargs)
