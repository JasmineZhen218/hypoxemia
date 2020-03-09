"""Contains functions for feature extraction.

Functions:
    linear_lstsq - Linear least-squares filtering
    linear_lstsq_deriv - Linear least-squares derivative approximation
    finite_approx_ewma_and_ewmvar - Finite-window approximations of exponentially weighted moving average and variance
    spectral_entropy - Spectral entropy of window of data
    moving_average - Moving average of data (not recommended)
    ema_and_emvar - Exponentially weighted moving average and variance
"""

import numpy as np
import math
import warnings
import matplotlib.pyplot as plt

from scipy import signal


def linear_lstsq(data, n):
    """Causally filter data using linear least-squares approximation.

    At the beginning of the data, where there is not enough previous data to do the calculation, the output is padded
    with nans. These weights are based on matching a pattern from a website.

    :param data: The data.
    :param n: The window of the linear least-squares filter.
    :return: The linear least-squares filtered data, padded at the beginning by nans.
    """
    weights = np.asarray([(6 * k + 4 * n - 2) / (n * (n + 1)) for k in range(- n + 1, 1)])
    return np.concatenate([np.ones(min(n - 1, len(data))) * np.nan, np.correlate(data, weights, mode='valid')])


def linear_lstsq_deriv(data, n):
    """Causally approximate the derivative data using linear least-squares.

    At the beginning of the data, where there is not enough previous data to do the calculation, the output is padded
    with nans. These weights are based on matching a pattern from a website.

    :param data: The data.
    :param n: The window of the linear least-squares filter.
    :return: The approximate derivative of the data, padded at the beginning by nans.
    """
    weights = np.asarray([(12 * k + 6 * n - 6) / ((n - 1) * n * (n + 1)) for k in range(- n + 1, 1)])
    return np.concatenate([np.ones(min(n - 1, len(data))) * np.nan, np.correlate(data, weights, mode='valid')])


def finite_approx_ewma_and_ewmvar(data, n):
    """Approximates the exponentially weighted moving average and variance using a window of finite length.

    :param data: The data to extract the features from.
    :param n: The window size
    :return: ewma, the approximate exponentially weighted moving average, and ewmvar, the approximate exponentially
        weighted moving variance.
    """
    alpha = 2 / (n + 1)
    weights = alpha * np.ones(n)
    for i in range(n):
        weights[n - i - 1] *= (1 - alpha) ** i
    weights = weights / np.sum(weights)
    ewma = np.concatenate([np.ones(n - 1) * np.nan, np.correlate(data, weights, mode='valid')])
    ewmvar = np.concatenate([np.ones(n - 1) * np.nan, np.correlate(np.square(data - ewma), weights, mode='valid')])
    return ewma, ewmvar


def spectral_entropy(data, n):
    """Computes the windowed real-time spectral entropy of the data.

    Note: This function does NOT detrend. Any detrending/debiasing must be done ahead of time.
    TODO: Ensure that this works properly.

    :param data: The data to extract the entropy from.
    :param n: The size of the window.
    :return: The spectral entropy.
    """
    num_nans = np.argwhere(~np.isnan(data))[0][0]
    f, t, Zxx = signal.stft(data[num_nans:],
                            window='blackmanharris',
                            nperseg=n,
                            noverlap=n - 1,
                            detrend=False,
                            boundary=None)
    power_spectra = np.square(np.absolute(Zxx))
    epsilon = 1e-30
    normed_ps = power_spectra / (np.sum(power_spectra, axis=0) + epsilon)
    return np.concatenate([np.ones(n - 1 + num_nans) * np.nan,
                           -np.sum(np.multiply(normed_ps, np.log2(normed_ps + epsilon)), axis=0) / np.log2(n)])


def spectral_entropy_detrend(data, filtered, n):
    return spectral_entropy(data - filtered, n)


def moving_average(data, n):
    """Compute the causal moving average of 1D data.

    At the beginning of the data, where there is not enough previous data to do the calculation, the output is padded
    with nans.

    NOTE: This function is NOT recommended for feature extraction, since it is linearly dependent with linear least
        squares filtering and linear least squares derivative approximation.

    :param data: The data.
    :param n: The window size of the moving average.
    :return: The moving average, padded with nans in the front.
    """
    warnings.warn("The moving average is not recommended for feature extraction.")
    return np.concatenate([np.ones(n - 1) * np.nan, np.convolve(data, np.ones(n), mode='valid') / n])


def ema_and_emvar(data, alpha):
    """Compute the exponentially weighted moving average and variance.

    This function handles nans in the input by carrying forward the mean and variance of the last non-nan data point as
    the most recent mean and variance. We might instead want to start over when a nan occurs - this is something to ask
    the engineering PIs about.

    NOTE: This function is NOT recommended for feature extraction, since it uses an unbounded window.

    :param data: The data.
    :param alpha: The parameter for exponential weighting.
    :return: ema, the exponentially weighted moving average, and emvar, the exponentially weighted moving variance.
    """
    warnings.warn("Unbounded exponentially weighted features are not recommended for feature extraction.")
    ema = np.zeros_like(data)
    emvar = np.zeros_like(data)
    ema[0] = data[0]
    last_valid_ema = ema[0]
    last_valid_emvar = emvar[0]
    for i, cur in enumerate(data[1:]):
        i = i + 1
        if math.isnan(cur):
            ema[i] = np.nan
            emvar[i] = np.nan
        else:
            delta = cur - last_valid_ema
            ema[i] = last_valid_ema + alpha * delta
            emvar[i] = (1 - alpha) * (last_valid_emvar + alpha * delta ** 2)
            last_valid_ema = ema[i]
            last_valid_emvar = emvar[i]
    return ema, emvar
