"""This implements a number of custom loss/metric functions for use in CLIPNET."""

import tensorflow as tf
from scipy.stats import spearmanr


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


def tf_spearmanr(x, y):
    return tf.py_function(
        spearmanr,
        [tf.cast(x, tf.float32), tf.cast(y, tf.float32)],
        Tout=tf.float32,
    )


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


def jaccard_distance(y_true, y_pred, smooth=100):
    """Calculates mean of Jaccard distance as a loss function"""
    y = tf.cast(y_true, tf.float32)
    intersection = tf.math.reduce_sum(y * y_pred)
    sum_ = tf.math.reduce_sum(y + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd = (1 - jac) * smooth
    return jd  # tf.reduce_mean(jd)
