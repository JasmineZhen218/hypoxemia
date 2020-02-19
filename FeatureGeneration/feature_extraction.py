import numpy as np
import json
import matplotlib.pyplot as plt
import math

from scipy import signal
from scipy.fftpack import fft


def ewma_vectorized_safe(data, alpha, row_size=None, dtype=None, order='C', out=None):
    """
    Copied from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    Reshapes data before calculating EWMA, then iterates once over the rows
    to calculate the offset without precision issues
    :param data: Input data, will be flattened.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param row_size: int, optional
        The row size to use in the computation. High row sizes need higher precision,
        low values will impact performance. The optimal value depends on the
        platform and the alpha being used. Higher alpha values require lower
        row size. Default depends on dtype.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    :return: The flattened result.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float
    else:
        dtype = np.dtype(dtype)

    row_size = int(row_size) if row_size is not None else get_max_row_size(alpha, dtype)

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
                       order='C', out=out_main_view)

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                        dtype=dtype, order='C', out=out[-trailing_n:])
    return out


def get_max_row_size(alpha, dtype=float):
    # Copied from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    assert 0. <= alpha < 1.
    # This will return the maximum row size possible on
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use np.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = np.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(np.log(epsilon)/np.log(1-alpha)) + 1


def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Copied from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None):
    """
    Copied from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out


def moving_average(data, n):
    """Compute the causal moving average of 1D data.

    At the beginning of the data, where there is not enough previous data to do the calculation, the output is padded
    with nans.

    :param data: The data.
    :param n: The window size of the moving average.
    :return: The moving average, padded with nans in the front.
    """
    return np.concatenate([np.ones(n - 1) * np.nan, np.convolve(data, np.ones(n), mode='valid') / n])


def ema_and_emvar(data, alpha):
    """Compute the exponentially weighted moving average and variance.

    This function handles nans in the input by carrying forward the mean and variance of the last non-nan data point as
    the most recent mean and variance. We might instead want to start over when a nan occurs - this is something to ask
    the engineering PIs about.
    TODO: Ask PIs about above.
    TODO: Make more efficient (vectorized).

    :param data: The data.
    :param alpha: The parameter for exponential weighting.
    :return: ema, the exponentially weighted moving average, and emvar, the exponentially weighted moving variance.
    """
    ema = np.zeros_like(data)
    emvar = np.zeros_like(data)
    ema[0] = data[0]
    last_valid_ema = ema[0]
    last_valid_emvar = emvar[0]
    for i, cur in enumerate(data[1:]):
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


def triangle(n):
    """Compute the nth triangle number; helper function for linear_lstsq."""
    sum = 0
    for i in range(n + 1):
        sum += i
    return sum


def triangles(n):
    """Compute the first n triangle numbers; helper function for tetrahedron."""
    triangles = np.zeros(n + 1)
    for i in range(n + 1):
        for j in range(i, n + 1):
            triangles[j] += i
    return triangles


def tetrahedron(n):
    """Compute the nth tetrahedral number; helper function for linear_lstsq_deriv."""
    sum = 0
    tris = triangles(n)
    for tri in tris:
        sum += tri
    return sum


def linear_lstsq(data, n):
    """Causally filter data using linear least-squares approximation.

    At the beginning of the data, where there is not enough previous data to do the calculation, the output is padded
    with nans. These weights are based on matching a pattern from a website.
    TODO: Prove that the weights are always correct.

    :param data: The data.
    :param n: The window of the linear least-squares filter.
    :return: The linear least-squares filtered data, padded at the beginning by nans.
    """
    tri = triangle(n)
    newest_weight = (2 * n - 1) / tri
    oldest_weight = - (n - 2) / tri
    increment = (newest_weight - oldest_weight) / (n - 1)
    weights = oldest_weight * np.ones(n) + increment * np.arange(0, n)
    return np.concatenate([np.ones(min(n - 1, len(data))) * np.nan, np.correlate(data, weights, mode='valid')])


def linear_lstsq_deriv(data, n):
    """Causally approximate the derivative data using linear least-squares.

    At the beginning of the data, where there is not enough previous data to do the calculation, the output is padded
    with nans. These weights are based on matching a pattern from a website.
    TODO: Prove that the weights are always correct.

    :param data: The data.
    :param n: The window of the linear least-squares filter.
    :return: The approximate derivative of the data, padded at the beginning by nans.
    """
    tetra = tetrahedron(n - 1)
    edge_weight = (n - 1) / tetra
    increment = (2 * edge_weight) / (n - 1)
    weights = - edge_weight * np.ones(n) + increment * np.arange(0, n)
    return np.concatenate([np.ones(min(n - 1, len(data))) * np.nan, np.correlate(data, weights, mode='valid')])


# def spectral_entropy(data, n):
#     f, t, Zxx = signal.stft(data, window='blackmanharris', nfft=n, detrend='linear')
#     power_spectrum = np.square(freq)
#     normed_ps = power_spectrum / np.sum(power_spectrum)
#     entropy = -np.sum(np.multiply(normed_ps, np.log2(normed_ps)) / np.log2(n))
#     return entropy


def extract_features(raw_name):
    """Extract additional features for a particular feature.

    Assumes that the current folder has a subfolder names "Raw" with the data in it and one called "Features" to which
    the features will be saved.

    :param raw_name: The name of the original feature.
    """
    save_path = raw_name + '_Features/'
    # save_path = 'Demo/'
    num_records = 0
    all_data = dict()
    with open('Raw/' + raw_name + '.json', 'r') as json_file:
        data = json.load(json_file)
        print('Finished Reading')
        num_entries = len(data)
        for name, raw_data in data.items():
            if num_records % 500 == 0:
                print('{} of {}'.format(num_records, num_entries))
            num_records += 1
            try:
                mean_5 = moving_average(raw_data, 5)
                mean_20 = moving_average(raw_data, 20)
                mean_60 = moving_average(raw_data, 60)
                # ewma_0_2 = ewma_vectorized_safe(raw_data, 0.2)
                # ewma_0_1 = ewma_vectorized_safe(raw_data, 0.1)
                # ewma_0_05 = ewma_vectorized_safe(raw_data, 0.05)
                # ewma_0_01 = ewma_vectorized_safe(raw_data, 0.01)
                ewma_0_2, ewmvar_0_2 = ema_and_emvar(raw_data, 0.2)
                ewma_0_1, ewmvar_0_1 = ema_and_emvar(raw_data, 0.1)
                ewma_0_05, ewmvar_0_05 = ema_and_emvar(raw_data, 0.05)
                ewma_0_01, ewmvar_0_01 = ema_and_emvar(raw_data, 0.01)
                lin_lstsq_5 = linear_lstsq(raw_data, 5)
                lin_lstsq_20 = linear_lstsq(raw_data, 20)
                lin_lstsq_60 = linear_lstsq(raw_data, 60)
                lin_lstsq_deriv_5 = linear_lstsq_deriv(raw_data, 5)
                lin_lstsq_deriv_20 = linear_lstsq_deriv(raw_data, 20)
                lin_lstsq_deriv_60 = linear_lstsq_deriv(raw_data, 60)
                spec_ent_20 = spectral_entropy(raw_data, 20)
                spec_ent_60 = spectral_entropy(raw_data, 60)
            except:
                continue
            dtype = np.float32
            data_2 = {
                'mean_5': mean_5.astype(dtype).tolist(),
                'mean_20': mean_20.astype(dtype).tolist(),
                'mean_60': mean_60.astype(dtype).tolist(),
                'ewma_0_2': ewma_0_2.astype(dtype).tolist(),
                'ewma_0_1': ewma_0_1.astype(dtype).tolist(),
                'ewma_0_05': ewma_0_05.astype(dtype).tolist(),
                'ewma_0_01': ewma_0_01.astype(dtype).tolist(),
                'ewmvar_0_2': ewmvar_0_2.astype(dtype).tolist(),
                'ewmvar_0_1': ewmvar_0_1.astype(dtype).tolist(),
                'ewmvar_0_05': ewmvar_0_05.astype(dtype).tolist(),
                'ewmvar_0_01': ewmvar_0_01.astype(dtype).tolist(),
                'lin_lstsq_5': lin_lstsq_5.astype(dtype).tolist(),
                'lin_lstsq_20': lin_lstsq_20.astype(dtype).tolist(),
                'lin_lstsq_60': lin_lstsq_60.astype(dtype).tolist(),
                'lin_lstsq_deriv_5': lin_lstsq_deriv_5.astype(dtype).tolist(),
                'lin_lstsq_deriv_20': lin_lstsq_deriv_20.astype(dtype).tolist(),
                'lin_lstsq_deriv_60': lin_lstsq_deriv_60.astype(dtype).tolist(),
                'spec_ent_20': spec_ent_20.astype(dtype).tolist(),
                'spec_ent_60': spec_ent_60.astype(dtype).tolist(),
            }
            all_data[name] = data_2
            # if name == 'p000052-2191-01-10-02-21n':
            #     with open(save_path + name + '.json', 'w') as savefile:
            #         json.dump(data_2, savefile)
            # with open(save_path + name + '.json', 'w') as savefile:
            #     json.dump(data_2, savefile)
    with open('Features/' + raw_name + '_Features.json', 'w') as savefile:
        print('About to write')
        json.dump(all_data, savefile)


def check_features():
    with open('Demo/p000052-2191-01-10-02-21n.json', 'r') as json_file:
        data = json.load(json_file)
        with open('SpO2_and_hypoxemia_labels/p000052-2191-01-10-02-21n.json', 'r') as json_file:
            data_2 = json.load(json_file)
            plt.figure()
            plt.title('SpO2 and Moving Averages')
            plt.plot(data_2['SpO2'])
            plt.plot(data['mean_5'])
            plt.plot(data['mean_20'])
            plt.plot(data['mean_60'])
            # plt.plot(data['lin_lstsq_5'])
            # plt.plot(data['lin_lstsq_20'])
            # plt.plot(data['lin_lstsq_60'])
            # plt.plot(data['lin_lstsq_deriv_5'])
            # plt.plot(data['lin_lstsq_deriv_20'])
            # plt.plot(data['lin_lstsq_deriv_60'])
            plt.xlabel('Time (min)')
            plt.legend(['SpO2', '5-min moving average', '20-min moving average', '60-min moving average'])
            plt.show()


if __name__ == '__main__':
    extract_features('SpO2')
    # check_features()
