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


def linear_lstsq(data, n, pos=None):
    """Causally filter data using linear least-squares approximation.

    At the beginning of the data, where there is not enough previous data to do the calculation, the output is padded
    with nans. These weights are based on matching a pattern from a website.

    :param data: The data.
    :param n: The window of the linear least-squares filter.
    :param pos: The position in the data at which to extract the feature. If None, extracts feature for whole sequence.
    :return: The linear least-squares filtered data, padded at the beginning by nans.
    """
    weights = np.asarray([(6 * k + 4 * n - 2) / (n * (n + 1)) for k in range(- n + 1, 1)])
    if pos is None:
        return np.concatenate([np.ones(min(n - 1, len(data))) * np.nan, np.correlate(data, weights, mode='valid')])
    else:
        if pos < n - 1:
            return np.nan
        return np.dot(weights, data[(pos - n + 1):(pos + 1)])


def linear_lstsq_deriv(data, n, pos=None):
    """Causally approximate the derivative data using linear least-squares.

    At the beginning of the data, where there is not enough previous data to do the calculation, the output is padded
    with nans. These weights are based on matching a pattern from a website.

    :param data: The data.
    :param n: The window of the linear least-squares filter.
    :param pos: The position in the data at which to extract the feature. If None, extracts feature for whole sequence.
    :return: The approximate derivative of the data, padded at the beginning by nans.
    """
    weights = np.asarray([(12 * k + 6 * n - 6) / ((n - 1) * n * (n + 1)) for k in range(- n + 1, 1)])
    if pos is None:
        return np.concatenate([np.ones(min(n - 1, len(data))) * np.nan, np.correlate(data, weights, mode='valid')])
    else:
        if pos < n - 1:
            return np.nan
        return np.dot(weights, data[(pos - n + 1):(pos + 1)])


def finite_approx_ewma_and_ewmvar(data, n, pos=None):
    """Approximates the exponentially weighted moving average and variance using a window of finite length.

    :param data: The data to extract the features from.
    :param n: The window size
    :param pos: The position in the data at which to extract the feature. If None, extracts feature for whole sequence.
    :return: ewma, the approximate exponentially weighted moving average, and ewmvar, the approximate exponentially
        weighted moving variance.
    """
    alpha = 2 / (n + 1)
    weights = alpha * np.ones(n)
    for i in range(n):
        weights[n - i - 1] *= (1 - alpha) ** i
    weights = weights / np.sum(weights)
    if pos is None:
        ewma = np.concatenate([np.ones(n - 1) * np.nan, np.correlate(data, weights, mode='valid')])
        ewmvar = np.concatenate([np.ones(n - 1) * np.nan, np.correlate(np.square(data - ewma), weights, mode='valid')])
        return ewma, ewmvar
    else:
        if pos < n - 1:
            return np.nan, np.nan
        elif pos < 2 * n - 2:
            return np.dot(weights, data[(pos - n + 1):(pos + 1)]), np.nan
        else:
            ewma = np.correlate(data[(pos - 2 * n + 2):(pos + 1)], weights, mode='valid')
            return ewma[-1], np.dot(weights, np.square(data[(pos - n + 1):(pos + 1)] - ewma))


def spectral_entropy(data, n, pos=None, detrend=False):
    """Computes the windowed real-time spectral entropy of the data.

    Note: This function does NOT debias.
    TODO: Ensure that this works properly.

    :param data: The data to extract the entropy from.
    :param n: The size of the window.
    :param pos: The position in the data at which to extract the feature. If None, extracts feature for whole sequence.
    :param detrend: Whether or not to detrend the data.
    :return: The spectral entropy.
    """
    if pos is None:
        nan_position = np.argwhere(~np.isnan(data))
        if len(nan_position) == 0:
            return np.nan * np.ones_like(data)
        num_nans = nan_position[0][0]
        try:
            f, t, Zxx = signal.stft(data[num_nans:],
                                    window='blackmanharris',
                                    nperseg=n,
                                    noverlap=n - 1,
                                    detrend=False,
                                    boundary=None)
        except:
            print("An error has occurred in spectral entropy calculation.")
            return np.ones_like(data) * np.nan
        power_spectra = np.square(np.absolute(Zxx))
        epsilon = 1e-30
        normed_ps = power_spectra / (np.sum(power_spectra, axis=0) + epsilon)
        return np.concatenate([np.ones(n - 1 + num_nans) * np.nan,
                               -np.sum(np.multiply(normed_ps, np.log2(normed_ps + epsilon)), axis=0) / np.log2(n)])
    else:
        if pos < n - 1:
            return np.nan
        window_data = np.copy(data[(pos - n + 1):(pos + 1)])
        if detrend:
            if pos < 2 * n - 2:
                return np.nan
            trend = linear_lstsq(data[(pos - 2 * n + 2):(pos + 1)], n)
            window_data -= trend[(-n):]
        f, t, Zxx = signal.stft(window_data,
                                window='blackmanharris',
                                nperseg=n,
                                noverlap=n - 1,
                                detrend=False,
                                boundary=None)
        power_spectra = np.square(np.absolute(Zxx))
        epsilon = 1e-30
        normed_ps = power_spectra / (np.sum(power_spectra, axis=0) + epsilon)
        return (-np.sum(np.multiply(normed_ps, np.log2(normed_ps + epsilon)), axis=0) / np.log2(n))[0]


def spectral_entropy_detrend(data, filtered, n):
    try:
        return spectral_entropy(data - filtered, n)
    except:
        return np.ones_like(data) * np.nan


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
