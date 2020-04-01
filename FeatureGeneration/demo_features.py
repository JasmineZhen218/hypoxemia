import matplotlib.pyplot as plt
import numpy as np

import scipy

from feature_extraction import (
    moving_average,
    ema_and_emvar,
    finite_approx_ewma_and_ewmvar,
    linear_lstsq,
    linear_lstsq_deriv,
    spectral_entropy,
    spectral_entropy_detrend
)


def test_features():
    with open('SpO2_and_hypoxemia_labels/p000052-2191-01-10-02-21n.json', 'r') as json_file:
        # data = json.load(json_file)
        # raw_data = data['SpO2'][:500]
        np.random.seed(0)
        # signal = np.sin(np.arange(0, (6 * np.pi), (2 * np.pi / 50)))
        # signal = np.sin(np.arange(0, (20 * np.pi), (2 * np.pi / 100))) + \
        #          2 * np.sin(np.arange(0, (4 * np.pi), (2 * np.pi / 500))) + \
        #          0.5 * np.sin(np.arange(0, (200 * np.pi), (2 * np.pi / 10)))
        signal = np.concatenate([np.sin(np.arange(0, (10 * np.pi), (2 * np.pi / 100))),
                                 np.sin(np.arange(0, (4 * np.pi), (2 * np.pi / 500)))])
        # signal = np.concatenate([np.zeros(50), np.ones(50)])
        # signal = np.concatenate([np.zeros(500), np.ones(500)])
        # signal = np.concatenate([np.arange(0, 100), 100 * np.ones(100)])
        # signal = np.concatenate([np.arange(0, 1000), 1000 * np.ones(1000)])
        # signal = np.concatenate([1 - np.exp(-np.arange(0, 10, 10 / 1000)), np.ones(1000)])
        # signal = 4 * np.arange(0, 1, 1 / 150) * (1 - np.arange(0, 1, 1 / 150))
        # # noise_std = 4 * np.arange(0, 1, 1 / 150) * (1 - np.arange(0, 1, 1 / 150))
        # noise_std = np.ones(150)
        # noise_std = np.concatenate([0.09 * np.ones(50), 0.49 * np.ones(50), 0.25 * np.ones(50)])
        # noise_std = np.concatenate([0.09 * np.ones(500), 1 * np.ones(500)])
        # noise_std = np.concatenate([0.09 * np.ones(200), 0.49 * np.ones(200), 0.25 * np.ones(200)])
        # noise = np.asarray([np.random.normal(0, i) for i in noise_std])
        # raw_data = signal + noise
        raw_data = signal
        # raw_data = noise

        # mean_5 = moving_average(raw_data, 5)
        # mean_20 = moving_average(raw_data, 20)
        # ewma_0_333, ewmvar_0_333 = ema_and_emvar(raw_data, 0.333)
        # ewma_0_095, ewmvar_0_095 = ema_and_emvar(raw_data, 0.095)
        # fin_ewma_5, fin_ewmvar_5 = finite_approx_ewma_and_ewmvar(raw_data, 5)
        # fin_ewma_20, fin_ewmvar_20 = finite_approx_ewma_and_ewmvar(raw_data, 20)
        lin_lst_sq_5 = linear_lstsq(raw_data, 5)
        lin_lst_sq_20 = linear_lstsq(raw_data, 20)
        lin_lst_sq_60 = linear_lstsq(raw_data, 60)
        # lin_lst_sq_deriv_5 = linear_lstsq_deriv(raw_data, 5)
        # lin_lst_sq_deriv_20 = linear_lstsq_deriv(raw_data, 20)
        spec_entr_5 = spectral_entropy(raw_data, 5)
        spec_entr_20 = spectral_entropy(raw_data, 20)
        spec_entr_60 = spectral_entropy(raw_data, 60)
        spec_entr_det_5 = spectral_entropy_detrend(raw_data, lin_lst_sq_5, 5)
        spec_entr_det_20 = spectral_entropy_detrend(raw_data, lin_lst_sq_20, 20)
        spec_entr_det_60 = spectral_entropy_detrend(raw_data, lin_lst_sq_60, 60)

        # plt.figure(1)
        # plt.plot(signal)
        # plt.title('Signal')
        # plt.xlabel('Time (min)')
        #
        # plt.figure(2)
        # plt.plot(noise)
        # plt.title('Noise')
        # plt.xlabel('Time (min)')
        #
        plt.figure(3)
        plt.plot(raw_data)
        plt.title('Combined Data')
        plt.xlabel('Time (min)')

        # plt.figure(4)
        # plt.plot(raw_data)
        # plt.plot(signal)
        # plt.plot(mean_5)
        # plt.plot(mean_20)
        # plt.title('Moving Average')
        # plt.xlabel('Time (min)')
        # plt.legend(['Input', 'Signal', 'n = 5', 'n = 20'])
        # # plt.legend(['Signal', 'n = 5', 'n = 20'])
        #
        # plt.figure(5)
        # # plt.plot(raw_data)
        # plt.plot(signal)
        # plt.plot(ewma_0_333)
        # plt.plot(ewma_0_095)
        # plt.title('Exponentially Weighted Moving Average')
        # plt.xlabel('Time (min)')
        # # plt.legend(['Input', 'Signal', 'alpha = 0.333 ewma', 'alpha = 0.095 ewma'])
        # plt.legend(['Signal', 'alpha = 0.333 ewma', 'alpha = 0.095 ewma'])
        #
        # plt.figure(6)
        # # plt.plot(raw_data)
        # plt.plot(signal)
        # plt.plot(fin_ewma_5)
        # plt.plot(fin_ewma_20)
        # plt.title('Finite Approximation of Exponentially Weighted Moving Average')
        # plt.xlabel('Time (min)')
        # # plt.legend(['Input', 'Signal', 'n = 5', 'n = 20'])
        # plt.legend(['Signal', 'n = 5', 'n = 20'])
        #
        # plt.figure(7)
        # plt.plot(raw_data)
        # plt.plot(signal)
        # plt.plot(ewma_0_333)
        # plt.plot(ewma_0_095)
        # plt.plot(fin_ewma_5)
        # plt.plot(fin_ewma_20)
        # plt.title('Approximation Comparison')
        # plt.xlabel('Time (min)')
        # plt.legend(['Input', 'Signal', 'Original, alpha = 0.333', 'Original, alpha = 0.095',
        #             'Approximation, n = 5', 'Approximation, n = 20'])
        # # plt.legend(['Signal', 'Original, alpha = 0.333', 'Original, alpha = 0.095',
        # #             'Approximation, n = 5', 'Approximation, n = 20'])
        #
        # plt.figure(8)
        # plt.plot(raw_data)
        # plt.plot(mean_20)
        # plt.plot(ewma_0_095)
        # plt.title('Effect of Exponential Weighting')
        # plt.xlabel('Time (min)')
        # plt.legend(['Signal', 'Mean, n = 50', 'EWMA, alpha = 0.095'])
        #
        # plt.figure(9)
        # plt.plot(raw_data)
        # plt.plot(signal)
        # plt.plot(lin_lst_sq_5)
        # plt.plot(lin_lst_sq_20)
        # plt.plot(lin_lst_sq_60)
        # plt.title('Linear Least-Squares Filtering')
        # plt.xlabel('Time (min)')
        # plt.legend(['Input', 'Signal', 'n = 5', 'n = 20', 'n = 60'])
        # # plt.legend(['Signal', 'n = 5', 'n = 20'])
        #
        # plt.figure(10)
        # plt.plot(raw_data)
        # # plt.plot(signal)
        # plt.plot(mean_5)
        # plt.plot(fin_ewma_5)
        # plt.plot(lin_lst_sq_5)
        # plt.title('Filtering Comparison, n = 5')
        # plt.xlabel('Time (min)')
        # # plt.legend(['Input', 'Signal', 'Moving Average', 'Approximate Exponential', 'Linear Least-Squares'])
        # # plt.legend(['Signal', 'Moving Average', 'Approximate Exponential', 'Linear Least-Squares'])
        # plt.legend(['Input', 'Moving Average', 'Approximate Exponential', 'Linear Least-Squares'])
        #
        # plt.figure(11)
        # plt.plot(raw_data)
        # # plt.plot(signal)
        # plt.plot(mean_20)
        # plt.plot(fin_ewma_20)
        # plt.plot(lin_lst_sq_20)
        # plt.title('Filtering Comparison, n = 20')
        # plt.xlabel('Time (min)')
        # # plt.legend(['Input', 'Signal', 'Moving Average', 'Approximate Exponential', 'Linear Least-Squares'])
        # # plt.legend(['Signal', 'Moving Average', 'Approximate Exponential', 'Linear Least-Squares'])
        # plt.legend(['Input', 'Moving Average', 'Approximate Exponential', 'Linear Least-Squares'])
        #
        # plt.figure(12)
        # plt.bar(range(-(5 - 1), 1), np.ones(5) / 5)
        # plt.title('Moving Average n = 5 Filter')
        #
        # plt.figure(13)
        # plt.bar(range(-(20 - 1), 1), np.ones(20) / 20)
        # plt.title('Moving Average n = 20 Filter')
        #
        # plt.figure(14)
        # n = 100
        # alpha = 0.333
        # weights = alpha * np.ones(n)
        # for i in range(n):
        #     weights[n - i - 1] *= (1 - alpha) ** i
        # plt.bar(range(-(n - 1), 1), weights)
        # plt.title('EWMA alpha = 0.333 Filter')

        # plt.figure(15)
        # n = 100
        # alpha = 0.095
        # weights = alpha * np.ones(n)
        # for i in range(n):
        #     weights[n - i - 1] *= (1 - alpha) ** i
        # plt.bar(range(-(n - 1), 1), weights)
        # plt.title('EWMA alpha = 0.095 Filter')

        # plt.figure(16)
        # n = 5
        # alpha = 2 / (n + 1)
        # weights = alpha * np.ones(n)
        # for i in range(n):
        #     weights[n - i - 1] *= (1 - alpha) ** i
        # weights = weights / np.sum(weights)
        # plt.bar(range(-(n - 1), 1), weights)
        # plt.title('Finite Approximate EWMA n = 5 Filter')
        #
        # plt.figure(17)
        # n = 20
        # alpha = 2 / (n + 1)
        # weights = alpha * np.ones(n)
        # for i in range(n):
        #     weights[n - i - 1] *= (1 - alpha) ** i
        # weights = weights / np.sum(weights)
        # plt.bar(range(-(n - 1), 1), weights)
        # plt.title('Finite Approximate EWMA n = 20 Filter')
        #
        # plt.figure(18)
        # n = 5
        # weights = np.asarray([(6 * k + 4 * n - 2) / (n * (n + 1)) for k in range(- n + 1, 1)])
        # plt.bar(range(-(n - 1), 1), weights)
        # plt.title('Linear Least-Squares n = 5 Filter')
        #
        # plt.figure(19)
        # n = 20
        # weights = np.asarray([(6 * k + 4 * n - 2) / (n * (n + 1)) for k in range(- n + 1, 1)])
        # plt.bar(range(-(n - 1), 1), weights)
        # plt.title('Linear Least-Squares n = 20 Filter')
        #
        # plt.figure(20)
        # plt.plot(np.gradient(signal))
        # plt.plot(lin_lst_sq_deriv_5)
        # plt.plot(lin_lst_sq_deriv_20)
        # plt.title('Linear Least-Squares Derivative Approximation')
        # plt.xlabel('Time (min)')
        # plt.legend(['Derivative of Signal', 'n = 5', 'n = 20'])

        # plt.figure(21)
        # plt.plot(np.square(noise_std))
        # plt.plot(ewmvar_0_333)
        # plt.plot(ewmvar_0_095)
        # plt.title('Exponentially Weighted Moving Variance')
        # plt.xlabel('Time (min)')
        # plt.legend(['Noise Distribution', 'alpha = 0.333', 'alpha = 0.095'])
        #
        # plt.figure(22)
        # # plt.plot(signal)
        # plt.plot(np.square(noise_std))
        # plt.plot(fin_ewmvar_5)
        # plt.plot(fin_ewmvar_20)
        # plt.title('Approximate Finite EWMV')
        # plt.xlabel('Time (min)')
        # plt.legend(['Noise Distribution', 'n = 5', 'n = 20'])

        plt.figure(23)
        # plt.plot(signal)
        # plt.plot(np.square(noise_std))
        plt.plot(spec_entr_5)
        plt.plot(spec_entr_20)
        plt.plot(spec_entr_60)
        plt.title('Spectral Entropy')
        plt.xlabel('Time (min)')
        plt.legend(['n = 5', 'n = 20', 'n = 60'])
        plt.ylim(0)

        plt.figure(24)
        # plt.plot(signal)
        # plt.plot(np.square(noise_std))
        plt.plot(spec_entr_det_5)
        plt.plot(spec_entr_det_20)
        plt.plot(spec_entr_det_60)
        plt.title('Spectral Entropy (Detrended)')
        plt.xlabel('Time (min)')
        plt.legend(['n = 5', 'n = 20', 'n = 60'])
        plt.ylim(0)

        plt.show()


def test_pos():
    np.random.seed(0)
    signal = np.random.rand(500)

    llsq_1 = linear_lstsq(signal, 15)
    llsq_2 = np.asarray([linear_lstsq(signal, 15, i) for i in range(500)])
    np.testing.assert_array_equal(llsq_1, llsq_2)

    llsqd_1 = linear_lstsq_deriv(signal, 15)
    llsqd_2 = np.asarray([linear_lstsq_deriv(signal, 15, i) for i in range(500)])
    np.testing.assert_array_equal(llsqd_1, llsqd_2)

    ewma_1, ewmvar_1 = finite_approx_ewma_and_ewmvar(signal, 15)
    ew_2 = np.asarray([finite_approx_ewma_and_ewmvar(signal, 15, i) for i in range(500)])
    ewma_2 = ew_2[:, 0]
    ewmvar_2 = ew_2[:, 1]
    np.testing.assert_array_equal(ewma_1, ewma_2)
    np.testing.assert_array_equal(ewmvar_1, ewmvar_2)

    spe_1 = spectral_entropy(signal, 15)
    spe_2 = np.asarray([spectral_entropy(signal, 15, i) for i in range(500)])
    np.testing.assert_array_equal(spe_1, spe_2)

    spe_det_1 = spectral_entropy_detrend(signal, linear_lstsq(signal, 15), 15)
    spe_det_2 = np.asarray([spectral_entropy(signal, 15, i, True) for i in range(500)])
    np.testing.assert_array_equal(spe_det_1, spe_det_2)

    print('All tests passed')


if __name__ == '__main__':
    # test_features()
    test_pos()
