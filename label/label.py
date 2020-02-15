# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 21:18:50 2020

@author: Jasmine
"""
import numpy as np
import matplotlib.pyplot as plt
import math

def nan_ratio(win):
    num=0
    for i in win:
        if math.isnan(i):
            num=num+1
    return num/len(win)

def hypo_event(sample):
    #len(sample)=5
    for i in sample:
        if i>92:
            return 0
        if math.isnan(i):
            return 0
    return 1
def event_within(win):
    for i in range(len(win)-5):
        sample=win[i:i+5]
        if hypo_event(sample)==1:
            return 1
    return 0

def Event(spo2):
    event=[0]*len(spo2)
    for i in range(len(event)-4):
        sample=spo2[i:i+5]
        if hypo_event(sample)==1:
            event[i:i+5]=[1]*5
            i=i+5
        else:
            event[i]=0
        event[len(event)-4:len(event)]=[event[len(event)-5]]*4
    return event
      
def Label(spo2):
    label=[0]*len(spo2)
    for i in range(len(label)-30):
        window30=spo2[i+1:i+31]
        if nan_ratio(window30)>0.2:
            label[i]=np.nan
        else:
            label[i]=event_within(window30)   
        label[len(label)-30:len(label)]=[np.nan]*30
    
    event=Event(spo2)
    label_usable=np.copy(np.asarray(label))
    label_usable[np.asarray(event)==1]=np.nan
    
    return label,label_usable

def summary(name,spo2):
    event=Event(spo2)
    label,label_usable=Label(spo2)
    
    
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
    