# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 08:05:21 2020

@author: Jasmine
"""

import datetime as dt
import pandas as pd
import numpy as np


# df1=pd.read_csv('/...csv') # sparse data to align
# df1.set_index('subject_id',inplace=True) 
# df2=pd.read_csv('meta_records.csv') # csv of information for each record

def align_sparse(ID,record_id,start,end,sparse):
    start_dt=dt.datetime.strptime(start,'%Y/%m/%d %H:%M')
    end_dt=dt.datetime.strptime(end,'%Y/%m/%d %H:%M')
    length=int((end_dt-start_dt).total_seconds()/60)
    indexes=[]
    value=[]
    for i, row in sparse.iterrows():
        charttime=row['charttime']
        charttime_dt=dt.datetime.strptime(charttime,'%Y-%m-%d %H:%M:%S')
        index_dt=charttime_dt-start_dt
        index=int(index_dt.total_seconds()/60)
        if (index>=-30) & (index<=length+30):
            indexes.append(index)
            value.append(row['value'])
    subject_id=[ID]*len(indexes)
    s=pd.DataFrame({
            'record_id':record_id,
            'subject_id':subject_id,
            'length':length,
            'time':indexes,
            'Analyzed FiO2':value      
                })
    return s

def s_to_df(s):
    df=pd.DataFrame(s.values.reshape(1,-1),columns=s.index)
    return df
    

sum=0
for index,row in df2.iterrows():
    ID=row['subject_id']
    if ID not in df1.index:
        continue
    record_id=row['record_id']
    start=row['start']
    end=row['end']
    sparse=df1.loc[ID]
    if type(sparse)==pd.Series:
        sparse=s_to_df(sparse)
    f=align_sparse(ID,record_id,start,end,sparse)
    if not f.empty:
        if sum==0:
            df=f
            sum=sum+1
        else:
            df=df.append(f)
            
df=df.sort_values(['subject_id','record_id','time'])

# df.to_csv("/...csv") # save the aligned sparse data



            

