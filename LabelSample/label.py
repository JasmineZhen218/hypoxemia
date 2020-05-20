# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 21:18:50 2020

@author: Jasmine
"""
"""
This is the code to create labels using SpO2 time-series data
1. Locate all hypoxemia events: 
   A continuous 5-minute low-spo2 is called one "event"
2. Create raw label:Check every point:
   Whether at least one hypoxemia event occurs in the next 30 minutes
3. Exclude some samples to get usable label
   Exclusion criteria:
       1. Exclude points when hypoxemia event occurs
       ---> Hypoxemia-free interval only
       2. hypoxemia-free interval shorter than 30 minutes
       (Close events should be merged as one)
"""


import numpy as np
import matplotlib.pyplot as plt
import math

def cut_nan_tail(spo2):
    """
    Help function to remove "nan tail"(all NaN at the end of series)
    input: 
        spo2: a minute-to minute spo2 series
    return: 
        new spo2 which nan tail is removed
    """
    if math.isnan(spo2[-1]):
        #print("yes")
        for i in range(len(spo2)-1,-1,-1):
            #print(i)
            if not(math.isnan(spo2[i])):
                #print(i)
                spo2=spo2[:i+1]
                #print(spo2)
                return spo2
    else:
        return spo2

def nan_ratio(win):
    """
    Funtion to measure the missing data portion in a series window
    input:
        win: a minute-to-minute spo2 window
    return:
        the portion of missing data
    
    """
    num=0
    for i in win:
        if math.isnan(i):
            num=num+1
    return num/len(win)

def hypo_event(sample):
    """
    function to judge whether a 5-minute spo2 interval is a hypoxemia event
    If all 5 measurements<= 92, return 1
    Otherwise, return 0
    input:
        5-minutes spo2 interval
    return:
        Flag to indicate whther this is a event
    """
    for i in sample:
        if i>92:
            return 0
        if math.isnan(i):
            return 0
    return 1
def event_within(win):
    """
    Function to judge whether at least one hypoxemia event occurs in a time window
    input: 
        win: a minute-to-minute spo2 window
    Return:
        Flag to indicate whether at least one hypoxemia event occurs in a time window
    """
    for i in range(len(win)-5):
        sample=win[i:i+5]
        if hypo_event(sample)==1:
            return 1
    return 0

def Event(spo2):
    """
    Function to locate hypoxemia event
    input:
        spo2:  a minute-to-minute spo2 series
    return:
        a flag series with the same length as input
        each point is 0 or 1, 1 indicate this time point is in a hypoxemia event, 0 indicate not
    """
    event=[0]*len(spo2)
    i=0
    while i<len(event)-4:
        sample=spo2[i:i+5]
        if hypo_event(sample)==1:
            event[i:i+5]=[1]*5
            i=i+5
        else:
            event[i]=0
            i+=1
    event[len(event)-4:]=[event[len(event)-5]]*4
    for i in range(len(spo2)):
        if math.isnan(spo2[i]):
            event[i]=np.nan
    return event
      
def Label(spo2):
    """
    Function to create raw labels
    input:
        spo2: a minute-to-minute spo2 series
    return:
        a series with the same length with input, but the last 30 points are nan
    """
    if len(spo2)<=30:
        return [np.nan]*len(spo2)
    label=[0]*len(spo2)
    for i in range(len(label)-30):
        window30=spo2[i+1:i+31]
        if nan_ratio(window30)>0.2:
            label[i]=np.nan
        else:
            label[i]=event_within(window30)   
    label[len(label)-30:len(label)]=[np.nan]*30
    return label

def hypoxemia_free_len(event):
    """
    Function to find the location and length of hypoxemia-free interval
    input:
        event: hypoxemia event series
    return:
        A list of dictionary, each dictionary indicate one hypoxemia-free interval's location and length
    """
    FLAG=0
    free=[]
    i=0
    accumulator=0
    while i<len(event)-1:
        for i in range(len(event)):
            if event[i]==1 and FLAG==0:
                free.append(
                        {'position':accumulator,
                         'length':i})
                FLAG=1
            if event[i]==0 and FLAG==1:
                #print(i)
                event=event[i:]
                accumulator+=i
                i=0
                FLAG=0
                break
    #print(i)
    print("Hypoxemia-free interval are locates at:")
    if event[i]==0:
        free.append(
                        {'position':accumulator,
                         'length':i})
    return free



def exclude_label(label,event,free):
    """
    Function to exclude some samples from label series
    input:
        label: raw spo2 label
        event: hypoxemia event
        free: list of hypoxemia-free interval list
    output:
        usable label which exclude some samples from raw labels
    
    exclusion creatia:
       1. Exclude points when hypoxemia event occurs
       ---> Hypoxemia-free interval only
       2. hypoxemia-free interval shorter than 30 minutes
       (Close events should be merged as one)
    """
    label_usable=np.copy(np.asarray(label))
    label_usable[np.asarray(event)==1]=np.nan
    for item in free:
        pos=item['position']
        length=item['length']
        if length<30:
            label_usable[pos:pos+length]=np.nan
    
    return label_usable

def usable_label(name,spo2):
    """
    a leading function to get usable labels from spo2 series
    input:
        name: record name p000032-12-01-16-13n(example)
        spo2: spo2 setries 
    return:
        usable labels series(unusable points are indicated as nan)
    """
    event=Event(spo2)
    #print(event)
    label=Label(spo2)
    #print(label)
    free=hypoxemia_free_len(event)
    label_usable=exclude_label(label,event,free)
    return {'name':name,'usable_label':label_usable}

def summary(name,spo2):
    """
    Visualization function to see labeling results
    input:
        name: record name p000032-12-01-16-13n(example)
        spo2: spo2 setries 
    Show:
        1 spo2 series
        2 hypoxemia event series
        3 Raw label series
        4 Usable label series
        
    reuturn:
        number of positive samples and negative samples in usable labels
    """
    #print(spo2)
    event=Event(spo2)
    #print(event)
    label=Label(spo2)
    #print(label)
    free=hypoxemia_free_len(event)
    for i in free:
        print(i)
    label_usable=exclude_label(label,event,free)
    
    
    print(name)
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, sharex=True)

    ax1.plot(spo2)
    ax1.set_title("SpO2")

    ax2.plot(event)
    ax2.set_title("Hypoxemia Event")

    ax3.plot(label)
    ax3.set_title("Label")

    ax4.plot(label_usable)
    ax4.set_title("Usable label")
    
    fig. tight_layout()

    positive=np.argwhere(label_usable==1)
    print("There are %d positive labels in all"%np.size(positive))

    negative=np.argwhere(label_usable==0)
    print("There are %d negative labels in all\n"%np.size(negative))
    
    return np.size(positive),np.size(negative)
    
# =============read in spo2 here================================================================
# import json
# with open("D:spo2_clean.json")as f:
#     dat=json.load(f)
# names=list(dat.keys())
# =============================================================================
name=names[18]
spo2=cut_nan_tail(dat[name])
summary(name,spo2)
