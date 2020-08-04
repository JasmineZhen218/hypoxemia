# -*- coding: utf-8 -*-
"""
Modified for ventilation alignment: 7/29/2020
Wen Shi
"""

import datetime as dt
import pandas as pd
import numpy as np
from datetime import timedelta


df1=pd.read_csv('noninvavent.csv') # noninvavent
#df1=pd.read_csv('nippvvent.csv') # nippv
#df1=pd.read_csv('target_inva_vent_sparse.csv') # invavent
df2=pd.read_csv('events.csv') # csv of information for each record
#%% store subid and corresponding stayid
table = pd.read_csv('D:\mcsv\ICUSTAYS.csv')
sub = table['SUBJECT_ID']
stay = table['ICUSTAY_ID']
staydic = {}
leng = sub.count()
for i in range(leng):
    subid = str(sub[i])
    stayid = str(stay[i])
    staydic[stayid] = subid   
#%% store subid into df1 according to staydic
df1stays = df1['icustay_id']
df1subs = []
for i in df1stays:
    df1subs.append(staydic[str(i)])
df1['sub_id'] = df1subs
#%% unique patient id in df2
patients = df2['patient_ID']
patients = np.unique(patients)
#%% Loop through each row of df1 and df2
ventcount = 0
totalventminutes = 0
ventrectable = pd.DataFrame({'subject_id':[],
                             'vent_start_time':[],
                             'vent_duration_min':[],
                             'vent_end_time':[]})
for index,row in df1.iterrows():
    subject = int(row['sub_id'])
    if subject not in patients:
        continue
    ventcount += 1
    ventstarttime = row['starttime']
    ############################Change when switching to invasive vent
    ventduration = row['ventduration']
    #ventduration = row['duration_hours']
    ventstarttime_dt = dt.datetime.strptime(ventstarttime, '%Y-%m-%d %H:%M:%S')
    #ventstarttime_dt = dt.datetime.strptime(ventstarttime, '%m/%d/%Y %H:%M')
    ventduration_minute = int(ventduration * 60)
    ventendtime_dt = ventstarttime_dt + timedelta(minutes = ventduration_minute)
    for rec_index,rec_row in df2.iterrows():
        rec_subject = int(rec_row['patient_ID'])
        if rec_subject != subject:
            continue
        rec_time = rec_row['record_name'][8:-1]
        rec_time_dt = dt.datetime.strptime(rec_time, '%Y-%m-%d-%H-%M')
        rec_start_interval = int(rec_row['start_interval'])
        rec_starttime_dt = rec_time_dt + timedelta(minutes = rec_start_interval)
        rec_duration = int(rec_row['duration_interval'])
        rec_endtime_dt = rec_starttime_dt + timedelta(minutes = rec_duration)
        # Calculate overlap between the two time period
        t = [(ventstarttime_dt,ventendtime_dt),(rec_starttime_dt,rec_endtime_dt)]
        overlap = (min(t[0][1],t[1][1]) - max(t[0][0], t[1][0]))
        if overlap.total_seconds() <= 0:
            continue
        ventrecstarttime = max(t[0][0], t[1][0])
        ventrecendtime = min(t[0][1],t[1][1])
        ventrecduration = int((ventrecendtime-ventrecstarttime).total_seconds()/60)
        totalventminutes += ventrecduration
        ventrectable = ventrectable.append({'subject_id':rec_subject,
                             'vent_start_time':ventrecstarttime,
                             'vent_duration_min':ventrecduration,
                             'vent_end_time':ventrecendtime}, ignore_index=True)
print(totalventminutes)
#%% Output CSV
ventrectable.to_csv('nippv_vent_events.csv', index=False, sep = ',')