#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:49:31 2020

@author: jaisi8631
"""

from ieeg.auth import Session
import pandas as pd
import pickle

def get_seizure_data(username, password, iEEG_filename, start_time_usec, stop_time_usec, outputfile):
    print("\n\nGetting data from iEEG.org:")
    print("iEEG_filename: {0}".format(iEEG_filename))
    print("start_time_usec: {0}".format(start_time_usec))
    print("stop_time_usec: {0}".format(stop_time_usec))
    print("Saving to: {0}".format(outputfile))
    start_time_usec = int(start_time_usec)
    stop_time_usec = int(stop_time_usec)
    duration = stop_time_usec - start_time_usec
    s = Session(username, password)
    ds = s.open_dataset(iEEG_filename)
    channels = list(range(len(ds.ch_labels)))
    data = ds.get_data(start_time_usec, duration, channels)
    
    df = pd.DataFrame(data, columns=ds.ch_labels)
    
    fs = ds.get_time_series_details(ds.ch_labels[0]).sample_rate # sample rate
    with open(outputfile, 'wb') as f: pickle.dump([df, fs], f)
    print("...done\n")       