# -*- coding: utf-8 -*-
# Clean time-series data
"""
Created on Sun Dec 29 11:42:59 2019

@author: Jasmine
"""

import pandas as pd
import numpy as np

# remove impossible data
def exclude_impo(value,up,low):
    '''
    Input:
        value is a 1D list saving time-series data
        up is the high limit
        low is the low limit
    Output:
        value is a ndarray saving the data with impossible data excluded
    '''
    value=np.array(value)
    value[value<=low]=np.NAN
    value[value>up]=up
    return value

# help function to get a mask of whether interpolate or not
def consecutive_pad(s):
    """
    input: 
        s: pd.series containing time-series data
    return:
        s4: a series with same size of s to indicate the number of 
        consecutive numeric values or consecutive NA values       
    """
    s1=s.notnull()
    s2=s1.shift()
    s3=s1 != s2
    s4=s3.cumsum()
    return s4

def mask_gap_df(df,threshold):
    """
    input:
        df: the Dataframe to fill
        threshold: the threshold of gap length could be filled
    return:
        mask: a bool type Dataframe with same size of df to indicate
        wether this NA value should be fill or not
    """
    mask=df.copy()
    for i in df.columns:
        s=df[i]
        dff=pd.DataFrame(s)
        dff['con_pad']=consecutive_pad(s)
        dff['ones']=[1]*len(s)
        mask[i]=(dff.groupby('con_pad')['ones'].transform('count')<threshold) | (dff[i].notnull())
    return mask

def mask_gap_s(s,threshold):
    """
    input:
        s: the Series to fill
        threshold: the threshold of gap length could be filled
    return:
        mask: a bool type Series with same size of df to indicate
        wether this NA value should be fill or not
    """
    dff=pd.DataFrame(s,columns=['record'])
    dff['con_pad']=consecutive_pad(s)
    dff['ones']=[1]*len(s)
    mask=(dff.groupby('con_pad')['ones'].transform('count')<threshold) | (dff['record'].notnull())
    return mask

# main function to clean a time-series data
def clean_and_fill(dic,up,low,threshold):
    """
    argument:
        dict:the dictionary loaded from json file
        up: the upper limit of possible value
        low: the low limit of possible value
        threshold: the threshold of gap length to fill
    return: 
        dict_clean: a dictionary containing cleaned and 
        interpolated time-series data
    """
    dict_clean={}
    keys=list(dic.keys())
    for i in keys:
        clean=exclude_impo(dic[i],up,low)
        s=pd.Series(clean)
        s1=s.fillna(method='ffill')
        s2=pd.Series(np.nan,range(len(s1)))
        mask=mask_gap_s(s,threshold)
        s2[mask]=s1
        dict_clean[i]=list(s2)
    return dict_clean

    
    