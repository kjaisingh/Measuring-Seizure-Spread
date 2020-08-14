#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:06:41 2020

@author: jaisi8631
"""

"""
PRIOR TO RUNNING THIS FILE:
1. Download dataset via get_iEEG_data.py.
2. 
    
    
"""

def create_dataset(data, start_time, end_time):
    dataset = []
    dataset_targets = []
    labels = pd.read_csv("labels/hup138-labels.csv", header = None)
    
    for column in data:
        
        print("Reading data for electrode " + column)
        
        col_list = data[column].tolist()
        col_data = labels.loc[labels[0] == column]
        
        col_start_time = col_data.iat[0, 1]
        col_end_time = col_data.iat[0, 2]
        
        if(col_start_time == '-' or col_end_time == '-'):
            col_start_time = int(start_time + 1)
            col_end_time = int(start_time + 1)
        else:
            col_start_time = int(col_start_time)
            col_end_time = int(col_end_time)
            
            
        
        for index in range(SEQUENCE_LEN, data.shape[0], STEP_SIZE):
            sequence = col_list[(index - SEQUENCE_LEN) : index]
            sequence = [[i] for i in sequence]
            dataset.append(sequence)
    
            sequence_end_time = start_time + (index * FS * 10)
            
            if(sequence_end_time >= col_start_time and 
               sequence_end_time <= col_end_time):
                dataset_targets.append(1)
            else:
                dataset_targets.append(0)
                
    dataset = np.array(dataset)
    dataset_targets = np.array(dataset_targets)
    
    return dataset, dataset_targets