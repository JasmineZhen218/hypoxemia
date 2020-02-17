# from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
import os
import shutil
import wfdb
import csv
import math
import json
import feather
import pandas as pd


def plot_spo2(signal, prune=False):
    pruned_signals = np.where(signal > 0.3, signal, np.nan)
    plt.figure()
    plt.plot(pruned_signals if prune else signal)
    plt.ylabel('SpO$_2$ (%)')
    plt.show()


def identify_spo2_in_numeric():
    path_to_records = 'D:/TeamCoolMonkey/NumericalData/'
    record_file = open('RECORDS-numerics.txt')
    lines = record_file.readlines()
    records = list()
    locations = list()
    for line in lines:
        break_index = line.find('/', 4)
        records.append(line[(break_index + 1):-1])
        locations.append(line[:(break_index + 1)])
    record_file.close()
    info_list = list()
    num_total = len(records)
    count = 0

    with open('spo2_records.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Subject ID', 'Record', 'Location', 'SpO2 Index'])
        csvfile.close()

    for record, location in zip(records, locations):
        count += 1
        if count % int(num_total / 100) == 0:
            print('{} of {}'.format(count, num_total))
            with open('spo2_records.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for info in info_list:
                    writer.writerow(info)
                csvfile.close()
                info_list = list()
        # sig_name = wfdb.rdrecord(record, pb_dir='mimic3wdb/matched/' + location).sig_name
        try:
            recording = wfdb.rdrecord(path_to_records + record)
        except:
            continue
        sig_name = recording.sig_name
        patient_id = int(recording.record_name[1:7])
        spo2_index = -1
        for i, name in enumerate(sig_name):
            if name == 'SpO2':
                spo2_index = i
        if spo2_index == -1:
            continue
        info_list.append([patient_id, record, location, spo2_index])

    with open('spo2_records.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for info in info_list:
            writer.writerow(info)
        csvfile.close()


def check_json():
    with open('SpO2_and_hypoxemia_labels_5_min/p062646-2168-06-08-19-50n.json', 'r') as json_file:
        data = json.load(json_file)
        plt.figure()
        plt.title('SpO2')
        plt.plot(data['SpO2'])
        plt.xlabel('Time (min)')
        plt.show()
        plt.figure()
        plt.title('Hypoxemia Labels')
        plt.plot(data['hypoxemia'])
        plt.xlabel('Time (min)')
        plt.show()
        plt.figure()
        plt.title('Prediction Labels')
        plt.plot(data['prediction_labels'])
        plt.xlabel('Time (min)')
        plt.show()
        print(data['fs'])
        print(data['base_time'])
        print(data['base_date'])


def rewrite_as_feather_file():
    with open('spo2_records.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        csv_lines = [row for row in csv_reader][1:]
        num_total = len(csv_lines)
        data_list = list()
        for i, row in enumerate(csv_lines[:10]):
            record = row[1]
            patient = int(record[1:7])
            with open('SpO2_and_hypoxemia_labels/' + record + '.txt', 'r') as json_file:
                data = json.load(json_file)
                data['patient'] = patient
                data['experiences_hypoxemia'] = np.any(np.asarray(data['hypoxemia']) == 1)
                data_list.append(data)
        df = pd.DataFrame(data_list)
        feather.write_dataframe(df, 'spo2_hypoxemia.feather')


def redo_hypoxemia_again():
    save_path = 'SpO2_and_hypoxemia_labels/'
    spo2_threshold = 30
    hypoxemia_threshold = 92
    prediction_time = 30
    num_records = 0

    hypoxemia_records = 0
    normal_timepoints = 0
    hypoxemia_timepoints = 0
    negative_timepoints = 0
    positive_timepoints = 0
    patients_hypoxemia = dict()
    hypoxemic_positive = 0
    training_positive = 0
    negative_negative = 0
    hypoxemic_negative = 0

    with open('SpO2.json', 'r') as json_file:
        data = json.load(json_file)
        for name, raw_spo2 in data.items():
            if num_records % 500 == 0:
                print(num_records)
            num_records += 1
            thresholded_spo2 = np.where(np.asarray(raw_spo2) < spo2_threshold, np.nan, raw_spo2)
            hypoxemia = np.where(np.isnan(thresholded_spo2), np.nan,
                                 np.where(np.asarray(thresholded_spo2) < hypoxemia_threshold, 1, 0))
            filter = np.ones(prediction_time)
            try:
                blurred = np.convolve(np.where(np.isnan(hypoxemia), 0, hypoxemia), filter)
            except:
                continue
            prediction_labels = np.where(blurred[prediction_time:] > 0, 1, 0)
            data_2 = {
                'SpO2': thresholded_spo2.astype(np.float32).tolist(),
                'hypoxemia': hypoxemia.astype(np.float32).tolist(),
                'prediction_labels': prediction_labels.astype(np.float32).tolist(),
                'fs': 1 / 60
            }
            with open(save_path + name + '.json', 'w') as savefile:
                json.dump(data_2, savefile)

            patient = int(name[1:7])
            num_hypoxemia = np.count_nonzero(hypoxemia)
            experiences_hypoxemia = num_hypoxemia > 0
            hypoxemia_records += 1 if experiences_hypoxemia else 0

            if patient in patients_hypoxemia:
                patients_hypoxemia[patient] = experiences_hypoxemia or patients_hypoxemia[patient]
            else:
                patients_hypoxemia[patient] = experiences_hypoxemia

            hypoxemia = hypoxemia[:-1]
            hypoxemic_positive += np.sum(np.where(hypoxemia * prediction_labels, 1, 0))
            training_positive += np.sum(np.where((1 - hypoxemia) * prediction_labels, 1, 0))
            negative_negative += np.sum(np.where((1 - hypoxemia) * (1 - prediction_labels), 1, 0))
            hypoxemic_negative += np.sum(np.where(hypoxemia * (1 - prediction_labels), 1, 0))

            hypoxemia_timepoints += num_hypoxemia
            normal_timepoints += hypoxemia.shape[0] - num_hypoxemia
            num_positive = np.count_nonzero(prediction_labels)
            positive_timepoints += num_positive
            negative_timepoints += prediction_labels.shape[0] - num_positive

    metadata = {
        'num_records': int(num_records),
        'hypoxemia_records': int(hypoxemia_records),
        'num_patients': len(patients_hypoxemia.keys()),
        'patients_hypoxemia': sum([1 if hypo else 0 for hypo in patients_hypoxemia.values()]),
        'normal_timepoints': int(normal_timepoints),
        'hypoxemia_timepoints': int(hypoxemia_timepoints),
        'negative_timepoints': int(negative_timepoints),
        'positive_timepoints': int(positive_timepoints),
        'hypoxemic_positive': int(hypoxemic_positive),
        'training_positive': int(training_positive),
        'negative_negative': int(negative_negative),
        'hypoxemic_negative': int(hypoxemic_negative),
    }
    with open('metadata_2.txt', 'w') as savefile:
        json.dump(metadata, savefile)


def redo_metadata():
    save_path = 'SpO2_and_hypoxemia_labels/'
    num_records = 0
    hypoxemia_records = 0
    normal_timepoints = 0
    hypoxemia_timepoints = 0
    negative_timepoints = 0
    positive_timepoints = 0
    patients_hypoxemia = dict()
    hypoxemic_positive = 0
    training_positive = 0
    training_negative = 0
    hypoxemic_negative = 0

    for entry in os.scandir('SpO2_and_hypoxemia_labels'):
        if num_records % 500 == 0:
            print(num_records)
        num_records += 1
        with open('SpO2_and_hypoxemia_labels/' + entry.name, 'r') as json_file:
            data = json.load(json_file)
            hypoxemia = np.asarray(data['hypoxemia'])
            prediction_labels = np.asarray(data['prediction_labels'])

            patient = int(entry.name[1:7])
            num_hypoxemia = np.count_nonzero(hypoxemia)
            experiences_hypoxemia = num_hypoxemia > 0
            hypoxemia_records += 1 if experiences_hypoxemia else 0
            if patient in patients_hypoxemia:
                patients_hypoxemia[patient] = experiences_hypoxemia or patients_hypoxemia[patient]
            else:
                patients_hypoxemia[patient] = experiences_hypoxemia

            hypoxemia = hypoxemia[:-1]

            for hypo, prediction in zip(hypoxemia, prediction_labels):
                if hypo > 0:
                    hypoxemia_timepoints += 1
                    if prediction > 0:
                        positive_timepoints += 1
                        hypoxemic_positive += 1
                    else:
                        negative_timepoints += 1
                        hypoxemic_negative += 1
                else:
                    normal_timepoints += 1
                    if prediction > 0:
                        positive_timepoints += 1
                        training_positive += 1
                    else:
                        negative_timepoints += 1
                        training_negative += 1

    metadata = {
        'num_records': int(num_records),
        'hypoxemia_records': int(hypoxemia_records),
        'num_patients': len(patients_hypoxemia.keys()),
        'patients_hypoxemia': sum([1 if hypo else 0 for hypo in patients_hypoxemia.values()]),
        'normal_timepoints': int(normal_timepoints),
        'hypoxemia_timepoints': int(hypoxemia_timepoints),
        'negative_timepoints': int(negative_timepoints),
        'positive_timepoints': int(positive_timepoints),
        'hypoxemic_positive': int(hypoxemic_positive),
        'training_positive': int(training_positive),
        'training_negative': int(training_negative),
        'hypoxemic_negative': int(hypoxemic_negative),
    }
    with open('metadata.txt', 'w') as savefile:
        json.dump(metadata, savefile)


def hypoxemia_duration_histogram():
    durations = list()
    num_records = 0
    counter = 0
    flag = False
    for entry in os.scandir('SpO2_and_hypoxemia_labels_5_min'):
        if num_records % 500 == 0:
            print(num_records)
        num_records += 1
        with open('SpO2_and_hypoxemia_labels_5_min/' + entry.name, 'r') as json_file:
            data = json.load(json_file)
            hypoxemia = np.asarray(data['hypoxemia'])
            for hypo in hypoxemia:
                if hypo > 0:
                    counter += 1
                    flag = True
                elif flag:
                    durations.append(counter)
                    if counter > 2000:
                        print(entry.name)
                    counter = 0
                    flag = False
    plt.figure()
    plt.hist(np.clip(durations, 1, 1000))
    plt.title('Hyoxemia Durations')
    plt.xlabel('Duration (min)')
    plt.ylabel('Hypoxemia Events')
    plt.show()
    plt.figure()
    plt.hist(np.clip(durations, 5, 60), bins=56)
    plt.title('Hypoxemia Durations (Clipped at 60 minutes)')
    plt.xlabel('Duration (min)')
    plt.ylabel('Number of Hypoxemia Events')
    plt.show()
    duration_hist = dict()
    for i in range(max(durations)):
        duration_hist[i + 1] = 0
    for duration in durations:
        duration_hist[duration] += 1
    with open('durations_5_min.txt', 'w') as savefile:
        json.dump(duration_hist, savefile)


def calculate_number_of_instances():
    with open('durations_5_min.txt', 'r') as json_file:
        data = json.load(json_file)
        sum = 0
        for _, value in data.items():
            sum += value
        print(sum)


def calculate_pie_numbers():
    with open('durations_5_min.txt', 'r') as json_file:
        data = json.load(json_file)
        print('sum 1: {}'.format(data[str(1)]))
        print('sum 2: {}'.format(data[str(2)]))
        print('sum 3-5: {}'.format(sum([data[str(i)] for i in range(3, 6)])))
        print('sum 6-10: {}'.format(sum([data[str(i)] for i in range(6, 11)])))
        print('sum 11-30: {}'.format(sum([data[str(i)] for i in range(11, 31)])))
        print('sum 31-60: {}'.format(sum([data[str(i)] for i in range(31, 61)])))
        print('sum 61-180: {}'.format(sum([data[str(i)] for i in range(61, 181)])))


def hypoxemia_5_min():
    save_path = 'SpO2_and_hypoxemia_labels_5_min/'
    spo2_threshold = 30
    hypoxemia_threshold = 92
    min_hypoxemia_time = 5
    prediction_time = 30

    num_records = 0
    hypoxemia_records = 0
    normal_timepoints = 0
    hypoxemia_timepoints = 0
    negative_timepoints = 0
    positive_timepoints = 0
    patients_hypoxemia = dict()
    hypoxemic_positive = 0
    training_positive = 0
    training_negative = 0
    hypoxemic_negative = 0

    with open('SpO2.json', 'r') as json_file:
        data = json.load(json_file)
        for name, raw_spo2 in data.items():
            if num_records % 500 == 0:
                print(num_records)
            num_records += 1
            clipped_spo2 = np.where(np.asarray(raw_spo2) < spo2_threshold, np.nan, raw_spo2)
            if clipped_spo2.shape[0] < min_hypoxemia_time:
                continue
            under_threshold = np.where(np.isnan(clipped_spo2), 0,
                                       np.where(np.asarray(clipped_spo2) < hypoxemia_threshold, 1, 0))
            centers_of_runs = np.convolve(under_threshold, np.ones(min_hypoxemia_time), mode='valid')
            spread_out = np.convolve(np.where(centers_of_runs == 5, 1, 0), np.ones(min_hypoxemia_time), mode='full')
            initial_hypoxemia = np.where(spread_out > 0, 1, 0)
            hypoxemia = np.where(np.isnan(clipped_spo2), np.nan, initial_hypoxemia)
            try:
                blurred = np.convolve(np.where(np.isnan(hypoxemia), 0, hypoxemia), np.ones(prediction_time))
            except:
                continue
            initial_prediction_labels = np.where(blurred[prediction_time:] > 0, 1, 0)
            prediction_labels = np.where(np.isnan(clipped_spo2[:-1]), np.nan, initial_prediction_labels)
            data_3 = {
                'SpO2': clipped_spo2.astype(np.float32).tolist(),
                'hypoxemia': hypoxemia.astype(np.float32).tolist(),
                'prediction_labels': prediction_labels.astype(np.float32).tolist(),
                'fs': 1 / 60,
            }
            with open(save_path + name + '.json', 'w') as savefile:
                json.dump(data_3, savefile)

            patient = int(name[1:7])
            num_hypoxemia = np.nansum(hypoxemia)
            experiences_hypoxemia = num_hypoxemia > 0
            hypoxemia_records += 1 if experiences_hypoxemia else 0
            if patient in patients_hypoxemia:
                patients_hypoxemia[patient] = experiences_hypoxemia or patients_hypoxemia[patient]
            else:
                patients_hypoxemia[patient] = experiences_hypoxemia

            hypoxemia = hypoxemia[:-1]

            for hypo, prediction in zip(hypoxemia, prediction_labels):
                if np.isnan(hypo) or np.isnan(prediction):
                    continue
                if hypo > 0:
                    hypoxemia_timepoints += 1
                    if prediction > 0:
                        positive_timepoints += 1
                        hypoxemic_positive += 1
                    else:
                        negative_timepoints += 1
                        hypoxemic_negative += 1
                else:
                    normal_timepoints += 1
                    if prediction > 0:
                        positive_timepoints += 1
                        training_positive += 1
                    else:
                        negative_timepoints += 1
                        training_negative += 1

    metadata = {
        'num_records': int(num_records),
        'hypoxemia_records': int(hypoxemia_records),
        'num_patients': len(patients_hypoxemia.keys()),
        'patients_hypoxemia': sum([1 if hypo else 0 for hypo in patients_hypoxemia.values()]),
        'normal_timepoints': int(normal_timepoints),
        'hypoxemia_timepoints': int(hypoxemia_timepoints),
        'negative_timepoints': int(negative_timepoints),
        'positive_timepoints': int(positive_timepoints),
        'hypoxemic_positive': int(hypoxemic_positive),
        'training_positive': int(training_positive),
        'training_negative': int(training_negative),
        'hypoxemic_negative': int(hypoxemic_negative),
    }
    with open('metadata_5_min.txt', 'w') as savefile:
        json.dump(metadata, savefile)


def compute_patient():
    num_records = 0
    hypoxemia_records = 0
    patients_hypoxemia = dict()

    for entry in os.scandir('SpO2_and_hypoxemia_labels_5_min'):
        if num_records % 500 == 0:
            print(num_records)
        num_records += 1
        with open('SpO2_and_hypoxemia_labels_5_min/' + entry.name, 'r') as json_file:
            data = json.load(json_file)
            hypoxemia = np.asarray(data['hypoxemia'])
            patient = int(entry.name[1:7])
            num_hypoxemia = np.nansum(hypoxemia)
            experiences_hypoxemia = num_hypoxemia > 0
            hypoxemia_records += 1 if experiences_hypoxemia else 0
            if patient in patients_hypoxemia:
                patients_hypoxemia[patient] = experiences_hypoxemia or patients_hypoxemia[patient]
            else:
                patients_hypoxemia[patient] = experiences_hypoxemia
    print('Hypoxemia records: {}'.format(int(hypoxemia_records)))
    print('Patients hypoxemia: {}'.format(sum([1 if hypo else 0 for hypo in patients_hypoxemia.values()])))


def create_static_hypoxemia_labels():
    labels = dict()
    num_records = 0
    for entry in os.scandir('SpO2_and_hypoxemia_labels_5_min'):
        if num_records % 500 == 0:
            print(num_records)
        num_records += 1
        with open('SpO2_and_hypoxemia_labels_5_min/' + entry.name, 'r') as json_file:
            data = json.load(json_file)
            hypoxemia = np.asarray(data['hypoxemia'])
            patient = int(entry.name[1:7])
            num_hypoxemia = np.nansum(hypoxemia)
            experiences_hypoxemia = True if num_hypoxemia > 0 else False
            if patient in labels:
                labels[patient] = experiences_hypoxemia or labels[patient]
            else:
                labels[patient] = experiences_hypoxemia
    with open('static_hypoxemia_labels.json', 'w') as savefile:
        json.dump(labels, savefile)
    df = pd.DataFrame(labels, index=[0])
    feather.write_dataframe(df, 'static_hypoxemia_labels.feather')


def check_static_hypoxemia_labels():
    with open('static_hypoxemia_labels.json', 'r') as json_file:
        data = json.load(json_file)
        true_count = 0
        false_count = 0
        for val in data.values():
            if val:
                true_count += 1
            else:
                false_count += 1
        print(true_count)
        print(false_count)


if __name__ == '__main__':
    # identify_spo2_in_numeric()
    # label_hypoxemia()
    # check_json()
    # rewrite_as_feather_file()
    # examine_data()
    # improve_metadata()
    # redo_hypoxemia()
    # redo_hypoxemia_again()
    # redo_metadata()
    # hypoxemia_duration_histogram()
    # calculate_number_of_instances()
    # calculate_pie_numbers()
    # hypoxemia_5_min()
    # compute_patient()
    create_static_hypoxemia_labels()
    # check_static_hypoxemia_labels()
