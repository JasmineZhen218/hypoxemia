#this program extract time-series data of different channels into json file
import numpy as np
import wfdb
from itertools import chain
import json
from collections import defaultdict
#read the file containing all record names
f = open('/Volumes/Elements/hypoxemia/RECORDS-numerics', 'r')
mylines = f.readlines()
line_num = len(mylines)
#read the file containing all the channel names you want to extract
name = open('name_sorted.txt','r')
channel_name = name.readline().strip()
while channel_name:
    headfold = []
    patients = []
    timestart = []
    feature_dict = defaultdict()
    for i in range(0,line_num):
        line = mylines[i].strip().split('/')
        headfold = '{}'.format(line[0])
        patients = '{}'.format(line[1])
        timestart = '{}'.format(line[2])
        file_path = '/Volumes/Elements/TeamCoolMonkey/NumericalData/' + timestart
        try:
            signals, fields = wfdb.rdsamp(file_path, channel_names=[channel_name])#need change
            if fields['sig_len'] <= 1:
                continue
            else:
                #convert data per second to data per minute using median
                if fields['fs'] == 0.0166666666667:
                    signals_list = list(chain(*signals))
                if fields['fs'] == 1:
                    min_len = int(fields['sig_len'] / 60)
                    signals = np.array(list(chain(*signals)))
                    signals_list = [0 for _ in range(min_len + 1)]
                    for k in range(0, min_len):
                        the_k_min = np.array(signals[60 * k:60 * (k + 1)])
                        mask = ~np.isnan(the_k_min)
                        signals_list[k] = np.median(the_k_min[mask])
                    the_min_rest = np.array(signals[60 * min_len:])
                    mask_rest = ~np.isnan(the_min_rest)
                    signals_list[min_len] = np.median(the_min_rest[mask_rest])
            feature_dict[timestart] = signals_list
        except Exception as e:
            print(file_path)
            print(e)
            pass
        if i % 1000 == 0:
            print(i)
    #remove '/' in channel name
    file_name = channel_name.replace('/','')
    with open(file_name + '.json', 'w') as yy:
        json.dump(feature_dict, yy)
    channel_name = name.readline().strip()

