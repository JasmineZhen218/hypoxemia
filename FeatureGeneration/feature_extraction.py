import numpy as np
import json
import matplotlib.pyplot as plt


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


# def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#     """
#     Modified from: https://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
#     Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
#     The Savitzky-Golay filter removes high frequency noise from data.
#     It has the advantage of preserving the original shape and
#     features of the signal better than other types of filtering
#     approaches, such as moving averages techniques.
#     Parameters
#     ----------
#     y : array_like, shape (N,)
#         the values of the time history of the signal.
#     window_size : int
#         the length of the window. Must be an odd integer number.
#     order : int
#         the order of the polynomial used in the filtering.
#         Must be less then `window_size` - 1.
#     deriv: int
#         the order of the derivative to compute (default = 0 means only smoothing)
#     Returns
#     -------
#     ys : ndarray, shape (N)
#         the smoothed signal (or it's n-th derivative).
#     Notes
#     -----
#     The Savitzky-Golay is a type of low-pass filter, particularly
#     suited for smoothing noisy data. The main idea behind this
#     approach is to make for each point a least-square fit with a
#     polynomial of high order over a odd-sized window centered at
#     the point.
#     Examples
#     --------
#     t = np.linspace(-4, 4, 500)
#     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
#     ysg = savitzky_golay(y, window_size=31, order=4)
#     import matplotlib.pyplot as plt
#     plt.plot(t, y, label='Noisy signal')
#     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
#     plt.plot(t, ysg, 'r', label='Filtered signal')
#     plt.legend()
#     plt.show()
#     References
#     ----------
#     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
#        Data by Simplified Least Squares Procedures. Analytical
#        Chemistry, 1964, 36 (8), pp 1627-1639.
#     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
#        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
#        Cambridge University Press ISBN-13: 9780521880688
#     """
#     import numpy as np
#     from math import factorial
#
#     try:
#         window_size = np.abs(np.int(window_size))
#         order = np.abs(np.int(order))
#     except ValueError:
#         raise ValueError("window_size and order have to be of type int")
#     if window_size % 2 != 1 or window_size < 1:
#         raise TypeError("window_size size must be a positive odd number")
#     if window_size < order + 2:
#         raise TypeError("window_size is too small for the polynomials order")
#     order_range = range(order+1)
#     half_window = (window_size -1) // 2
#     # precompute coefficients
#     b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
#     m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
#     # pad the signal at the extremes with
#     # values taken from the signal itself
#     firstvals = [y[0] for _ in range(half_window)]
#     lastvals = [y[-1] for _ in range(half_window)]
#     y = np.concatenate([firstvals, y, lastvals])
#     return np.convolve( m[::-1], y, mode='valid')


def triangle(n):
    sum = 0
    for i in range(n + 1):
        sum += i
    return sum


def triangles(n):
    triangles = np.zeros(n + 1)
    for i in range(n + 1):
        for j in range(i, n + 1):
            triangles[j] += i
    return triangles


def tetrahedron(n):
    sum = 0
    tris = triangles(n)
    for tri in tris:
        sum += tri
    return sum


def linear_lstsq(data, n):
    tri = triangle(n)
    newest_weight = (2 * n - 1) / tri
    oldest_weight = - (n - 2) / tri
    increment = (newest_weight - oldest_weight) / (n - 1)
    weights = oldest_weight * np.ones(n) + increment * np.arange(0, n)
    return np.concatenate([np.ones(min(n - 1, len(data))) * np.nan, np.correlate(data, weights, mode='valid')])


def linear_lstsq_deriv(data, n):
    tetra = tetrahedron(n - 1)
    edge_weight = (n - 1) / tetra
    increment = (2 * edge_weight) / (n - 1)
    weights = - edge_weight * np.ones(n) + increment * np.arange(0, n)
    return np.concatenate([np.ones(min(n - 1, len(data))) * np.nan, np.correlate(data, weights, mode='valid')])


def extract_features(raw_name):
    # save_path = raw_name + '_Features/'
    save_path = 'Demo/'
    num_records = 0
    all_data = dict()
    with open('Raw/' + raw_name + '.json', 'r') as json_file:
        data = json.load(json_file)
        print('Finished Reading')
        num_entries = len(data)
        for name, raw_spo2 in data.items():
            if num_records % 500 == 0:
                print('{} of {}'.format(num_records, num_entries))
            num_records += 1
            try:
                mean_5 = np.concatenate([np.ones(4) * np.nan, np.convolve(raw_spo2, np.ones(5), mode='valid') / 5])
                mean_20 = np.concatenate([np.ones(19) * np.nan, np.convolve(raw_spo2, np.ones(20), mode='valid') / 20])
                mean_60 = np.concatenate([np.ones(59) * np.nan, np.convolve(raw_spo2, np.ones(60), mode='valid') / 60])
                ewma_0_2 = ewma_vectorized_safe(raw_spo2, 0.2)
                ewma_0_1 = ewma_vectorized_safe(raw_spo2, 0.1)
                ewma_0_05 = ewma_vectorized_safe(raw_spo2, 0.05)
                ewma_0_01 = ewma_vectorized_safe(raw_spo2, 0.01)
                lin_lstsq_5 = linear_lstsq(raw_spo2, 5)
                lin_lstsq_20 = linear_lstsq(raw_spo2, 20)
                lin_lstsq_60 = linear_lstsq(raw_spo2, 60)
                lin_lstsq_deriv_5 = linear_lstsq_deriv(raw_spo2, 5)
                lin_lstsq_deriv_20 = linear_lstsq_deriv(raw_spo2, 20)
                lin_lstsq_deriv_60 = linear_lstsq_deriv(raw_spo2, 60)
                # sg_5_3 = savitzky_golay(raw_spo2, window_size=5, order=3, deriv=0)
                # sg_11_7 = savitzky_golay(raw_spo2, window_size=11, order=7, deriv=0)
                # sg_31_15 = savitzky_golay(raw_spo2, window_size=31, order=15, deriv=0)
                # sg_deriv_5_3 = savitzky_golay(raw_spo2, window_size=5, order=3, deriv=1)
                # sg_deriv_11_7 = savitzky_golay(raw_spo2, window_size=11, order=7, deriv=1)
                # sg_deriv_31_15 = savitzky_golay(raw_spo2, window_size=31, order=15, deriv=1)
                # sg_deriv2_5_3 = savitzky_golay(raw_spo2, window_size=5, order=3, deriv=2)
                # sg_deriv2_11_7 = savitzky_golay(raw_spo2, window_size=11, order=7, deriv=2)
                # sg_deriv2_31_15 = savitzky_golay(raw_spo2, window_size=31, order=15, deriv=2)
            except:
                continue
            data_2 = {
                'mean_5': mean_5.astype(np.float32).tolist(),
                'mean_20': mean_20.astype(np.float32).tolist(),
                'mean_60': mean_60.astype(np.float32).tolist(),
                'ewma_0_2': ewma_0_2.astype(np.float32).tolist(),
                'ewma_0_1': ewma_0_1.astype(np.float32).tolist(),
                'ewma_0_05': ewma_0_05.astype(np.float32).tolist(),
                'ewma_0_01': ewma_0_01.astype(np.float32).tolist(),
                'lin_lstsq_5': lin_lstsq_5.astype(np.float32).tolist(),
                'lin_lstsq_20': lin_lstsq_20.astype(np.float32).tolist(),
                'lin_lstsq_60': lin_lstsq_60.astype(np.float32).tolist(),
                'lin_lstsq_deriv_5': lin_lstsq_deriv_5.astype(np.float32).tolist(),
                'lin_lstsq_deriv_20': lin_lstsq_deriv_20.astype(np.float32).tolist(),
                'lin_lstsq_deriv_60': lin_lstsq_deriv_60.astype(np.float32).tolist(),
                # 'sg_5_3': sg_5_3.astype(np.float32).tolist(),
                # 'sg_11_7': sg_11_7.astype(np.float32).tolist(),
                # 'sg_31_15': sg_31_15.astype(np.float32).tolist(),
                # 'sg_deriv_5_3': sg_deriv_5_3.astype(np.float32).tolist(),
                # 'sg_deriv_11_7': sg_deriv_11_7.astype(np.float32).tolist(),
                # 'sg_deriv_31_15': sg_deriv_31_15.astype(np.float32).tolist(),
                # 'sg_deriv2_5_3': sg_deriv2_5_3.astype(np.float32).tolist(),
                # 'sg_deriv2_11_7': sg_deriv2_11_7.astype(np.float32).tolist(),
                # 'sg_deriv2_31_15': sg_deriv2_31_15.astype(np.float32).tolist(),
            }
            all_data[name] = data_2
            if name == 'p000052-2191-01-10-02-21n':
                with open(save_path + name + '.json', 'w') as savefile:
                    json.dump(data_2, savefile)
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
    # extract_features('SpO2')
    check_features()
