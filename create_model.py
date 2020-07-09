#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:58:18 2020

@author: jaisi8631
"""

import pickle
import pandas as pd
import numpy as np

start_time = 415839606029
end_time = 416311906098
SEQUENCE_LEN = 2048

with open("../datasets/hup138.pickle", 'rb') as f: data, fs = pickle.load(f)

labels = pd.read_csv("../datasets/hup138-labels.csv", header = None)
labels_list = labels[0].tolist()
data = data[data.columns.intersection(labels_list)]

timestamps = []
for i in range(0, data.shape[0]):
    timestamps.append(start_time + (i * fs))
    

# training dataset dimensions should be (# samples, 2048, 1)
train = []
# training target dataset dimensions should be (# samples)
# 0 indicates non-seizure, 1 indicates seizure
train_targets = []

for column in data:
    
    print("Currently on Column: " + column)
    
    col_list = data[column].tolist()
    col_data = labels.loc[labels[0] == column]

    col_start_time = int(col_data.iat[0, 1])
    col_end_time = int(col_data.iat[0, 2])
    
    for index in range(SEQUENCE_LEN, data.shape[0], 16):
        
        sequence = col_list[(index - SEQUENCE_LEN):index]
        sequence = [[i] for i in sequence]
        train.append(sequence)

        sequence_end_time = col_list[index]
        if(sequence_end_time >= col_end_time):
            train_targets.append(1)
        else:
            train_targets.append(1)
        