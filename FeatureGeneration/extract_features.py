from feature_extraction import (
    finite_approx_ewma_and_ewmvar,
    linear_lstsq,
    linear_lstsq_deriv,
    spectral_entropy,
    spectral_entropy_detrend
)
import csv
import json
import numpy as np


writer = csv.writer(open('train-10min_pulse_2.csv', 'w'), lineterminator='\n')
reader = csv.reader(open('train-10min.csv', newline=''), delimiter=' ', quotechar='|')
with open('Raw/Pulse_clean.json', 'r') as json_file:
    data = json.load(json_file)
    header = next(reader)
    header = header[0].split(',')
    header = header + ['pulse']
    # header = header + ['pulse', 'pulse_ewma_5', 'pulse_ewmvar_5', 'pulse_llsq_5', 'pulse_llsqd_5', 'pulse_spe_5',
    #                    'pulse_spe_det_5', 'pulse_ewma_20', 'pulse_ewmvar_20', 'pulse_llsq_20', 'pulse_llsqd_20',
    #                    'pulse_spe_20', 'pulse_spe_det_20', 'pulse_ewma_60', 'pulse_ewmvar_60', 'pulse_llsq_60',
    #                    'pulse_llsqd_60', 'pulse_spe_60', 'pulse_spe_det_60']
    writer.writerow(header)
    for row in reader:
        row_info = row[0].split(',')
        try:
            raw_data = data[row_info[2]]
        except KeyError:
            writer.writerow(row_info + [np.nan for _ in range(19)])
            continue
        position = int(row_info[4])
        row_info = row_info + [raw_data[position]]
        writer.writerow(row_info)  # Remove
        # try:
        #     if position >= 4:
        #         window_5 = raw_data[(position - 4):(position + 1)]
        #         llsqd_5 = linear_lstsq_deriv(window_5, 5)
        #         spe_5 = spectral_entropy(window_5, 5)
        #         if position >= 9:
        #             window_5_wider = raw_data[(position - 9):(position + 1)]
        #             ewma_5, ewmvar_5 = finite_approx_ewma_and_ewmvar(window_5_wider, 5)
        #             llsq_5 = linear_lstsq(window_5_wider, 5)
        #             spe_det_5 = spectral_entropy_detrend(window_5_wider, llsq_5, 5)
        #         else:
        #             ewma_5, ewmvar_5 = finite_approx_ewma_and_ewmvar(window_5, 5)
        #             llsq_5 = linear_lstsq(window_5, 5)
        #         row_info = row_info + [ewma_5[-1], ewmvar_5[-1], llsq_5[-1], llsqd_5[-1], spe_5[-1]]
        #         if position >= 9:
        #             row_info = row_info + [spe_det_5[-1]]
        #         else:
        #             row_info = row_info + [np.nan]
        #     else:
        #         row_info = row_info + [np.nan for _ in range(6)]
        #     if position >= 19:
        #         window_20 = raw_data[(position - 19):(position + 1)]
        #         llsqd_20 = linear_lstsq_deriv(window_20, 20)
        #         spe_20 = spectral_entropy(window_20, 20)
        #         if position >= 39:
        #             window_20_wider = raw_data[(position - 39):(position + 1)]
        #             ewma_20, ewmvar_20 = finite_approx_ewma_and_ewmvar(window_20_wider, 20)
        #             llsq_20 = linear_lstsq(window_20_wider, 20)
        #             spe_det_20 = spectral_entropy_detrend(window_20_wider, llsq_20, 20)
        #         else:
        #             ewma_20, ewmvar_20 = finite_approx_ewma_and_ewmvar(window_20, 20)
        #             llsq_20 = linear_lstsq(window_20, 20)
        #         row_info = row_info + [ewma_20[-1], ewmvar_20[-1], llsq_20[-1], llsqd_20[-1], spe_20[-1]]
        #         if position >= 39:
        #             row_info = row_info + [spe_det_20[-1]]
        #         else:
        #             row_info = row_info + [np.nan]
        #     else:
        #         row_info = row_info + [np.nan for _ in range(6)]
        #     if position >= 59:
        #         window_60 = raw_data[(position - 59):(position + 1)]
        #         llsqd_60 = linear_lstsq_deriv(window_60, 60)
        #         spe_60 = spectral_entropy(window_60, 60)
        #         if position >= 119:
        #             window_60_wider = raw_data[(position - 119):(position + 1)]
        #             ewma_60, ewmvar_60 = finite_approx_ewma_and_ewmvar(window_60_wider, 60)
        #             llsq_60 = linear_lstsq(window_60_wider, 60)
        #             spe_det_60 = spectral_entropy_detrend(window_60_wider, llsq_60, 60)
        #         else:
        #             ewma_60, ewmvar_60 = finite_approx_ewma_and_ewmvar(window_60, 60)
        #             llsq_60 = linear_lstsq(window_60, 60)
        #         row_info = row_info + [ewma_60[-1], ewmvar_60[-1], llsq_60[-1], llsqd_60[-1], spe_60[-1]]
        #         if position >= 119:
        #             row_info = row_info + [spe_det_60[-1]]
        #         else:
        #             row_info = row_info + [np.nan]
        #     else:
        #         row_info = row_info + [np.nan for _ in range(6)]
        #     writer.writerow(row_info)
        # except:
        #     print("One of the other errors")
        #     writer.writerow(row_info + [np.nan for _ in range(19)])
