# epistasis.py
# Adam He <adamyhe@gmail.com>

"""
Calculate Deep Feature Interaction Maps (DFIM) using shap.DeepExplainer as the
attribution method.
"""

import numpy as np
import tqdm

from . import utils


def get_most_extreme(arr, axis=None):
    """
    Returns the most extreme value in a tensor along a given dimension.

    Parameters
    ----------
    arr : np.array, shape (..., D)
        The tensor to find the most extreme value in.
    dim : int
        The dimension along which to find the most extreme value.

    Returns
    -------
    np.array, shape (...)
        The most extreme value in the tensor along the given dimension.
    """
    tmax = arr.max(axis=axis)[0]
    tmin = arr.min(axis=axis)[0]
    return np.where(-tmin > tmax, tmin, tmax)


def dfim(explainers, major_seq, start, stop, check_additivity=True, silence=False):
    """
    Calculate Deep Feature Interaction Maps (DFIM) using shap.DeepExplainer as the
    attribution method.

    Parameters
    ----------
    explainers : list
        List of shap.DeepExplainer objects
    major_seq : str
        The sequence to calculate DFIM scores for.
    start : int
        The start position of where to calculate DFIM.
    stop : int
        The stop position of where to calculate DFIM.
    check_additivity : bool, optional
        Whether to check for additivity, by default True.
    silence : bool, optional
        Whether to silence tqdm, by default False

    Returns
    -------
    np.array, shape (stop - start, len(alphabet) - 1, len(sequence), len(alphabet))
        The DFIM scores for each position.
    """
    major_twohot = np.expand_dims(utils.TwoHotDNA(major_seq).twohot, axis=0)
    dfim_range = list(range(start, stop))
    major_shap = np.array(
        [
            explainer.shap_values(major_twohot, check_additivity=check_additivity)[0]
            for explainer in explainers
        ]
    ).mean(axis=0)
    mutations_per_pos = [utils.get_mut_bases(major_seq[i]) for i in dfim_range]
    dfim = []
    for i in tqdm.trange(len(mutations_per_pos), disable=silence):
        muts = mutations_per_pos[i]
        mut_seqs = [
            major_seq[: dfim_range[i]] + mut + major_seq[dfim_range[i] + 1 :]
            for mut in muts
        ]
        mut_twohot = np.array([utils.TwoHotDNA(mut_seq).twohot for mut_seq in mut_seqs])
        mut_shap = np.array(
            [
                explainer.shap_values(mut_twohot, check_additivity=check_additivity)[0]
                for explainer in explainers
            ]
        ).mean(axis=0)
        fis = major_shap * major_twohot / 2 - mut_shap * mut_twohot / 2
        dfim.append(fis)
    return np.array(dfim)
