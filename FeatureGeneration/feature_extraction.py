import numpy as np
import json
import matplotlib.pyplot as plt
import math

from scipy import signal
from scipy.fftpack import fft


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
    TODO: Ask PIs about nan issue above.

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


# def spectral_entropy(data, nfft):
#     f, t, Zxx = signal.stft(data, window='blackmanharris', nperseg=nfft, detrend='linear')
#     power_spectra = np.square(np.absolute(Zxx))
#     normed_ps = power_spectrum / np.sum(power_spectrum)
#     entropy = -np.sum(np.multiply(normed_ps, np.log2(normed_ps)) / np.log2(nfft))
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
                # spec_ent_20 = spectral_entropy(raw_data, 20)
                # spec_ent_60 = spectral_entropy(raw_data, 60)
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
                # 'spec_ent_20': spec_ent_20.astype(dtype).tolist(),
                # 'spec_ent_60': spec_ent_60.astype(dtype).tolist(),
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


def test_features():
    with open('SpO2_and_hypoxemia_labels/p000052-2191-01-10-02-21n.json', 'r') as json_file:
        data = json.load(json_file)
        raw_data = data['SpO2'][:500]
        # lin_lst_sq = linear_lstsq(raw_data, 5)
        # lin_lst_sq = linear_lstsq(raw_data, 10)
        # lin_lst_sq = linear_lstsq(raw_data, 13)
        # lin_lst_sq_deriv = linear_lstsq_deriv(raw_data, 5)
        # lin_lst_sq_deriv = linear_lstsq_deriv(raw_data, 10)
        # lin_lst_sq_deriv = linear_lstsq_deriv(raw_data, 13)
        ewma_0_2, ewmvar_0_2 = ema_and_emvar(raw_data, 0.2)
        ewma_0_05, ewmvar_0_05 = ema_and_emvar(raw_data, 0.05)

        plt.figure()
        plt.plot(raw_data)
        plt.plot(ewma_0_2)
        plt.plot(ewma_0_05)
        plt.xlabel('Time (min)')
        plt.legend(['SpO2', 'alpha = 0.2 moving average', 'alpha = 0.05 moving average'])
        plt.show()


if __name__ == '__main__':
    # extract_features('SpO2')
    test_features()
