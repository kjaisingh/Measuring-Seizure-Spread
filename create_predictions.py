#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 15:06:41 2020

@author: jaisi8631
"""


"""
RUNNING THIS FILE FOR DATASET WITH IDENTIFICATION NUMBER 'ID':
1. Download datasets via get_iEEG_data.py, and name them: 
   hupID-interictal.pickle and hupID-ictal.pickle
2. Create csv file with labels, and name it: hupID-labels.csv
3. 

"""
# ------------------
# REGULAR IMPORTS
# ------------------
import pickle
import pandas as pd
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV

import keras
from keras.utils.vis_utils import plot_model
from keras.models import Model 
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier


# ------------------
# LSTM IMPORTS
# ------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# ------------------
# CNN IMPORTS
# ------------------
from keras.optimizers import SGD
from keras.layers import Reshape
from keras.layers import Conv1D
from keras.layers import InputLayer
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D


# ------------------
# CONSTANTS
# ------------------
START_TIME_INTERICTAL = 356850680000
END_TIME_INTERICTAL = 356903099171
START_TIME_ICTAL = 402704260829
END_TIME_ICTAL = 402756680000
FS = 1024
DOWN_SAMPLE_FACTOR = 10
STEP_SIZE = 256
SEQUENCE_LEN = 1024

PATH_INTERICTAL = "../datasets/hup172-interictal.pickle"
PATH_ICTAL = "../datasets/hup172-ictal.pickle"
PATH_LABELS = "labels/hup172-labels.csv"
PATH_MODEL = 'models/eeg-model-cnn-wavenet'


# ------------------
# LOAD DATA
# ------------------
def get_data(path):
    with open(path, 'rb') as f: data, fs = pickle.load(f)
    labels = pd.read_csv(PATH_LABELS, header = None)
    labels_list = labels[0].tolist()
    data = data[data.columns.intersection(labels_list)]
    return data

def iEEG_data_filter(data, fs, cutoff1, cutoff2, notch):
	column_names = data.columns
	data = np.array(data)
	number_of_channels = data.shape[1]
    
    # Cut-off frequency of the filter
	fc = np.array([cutoff1, cutoff2])
    # Normalize the frequency
	w = fc / np.array([(fs / 2), (fs / 2)])  
    
	b, a = signal.butter(4, w, 'bandpass')
	filtered = np.zeros(data.shape)
	for i in np.arange(0,number_of_channels):
		filtered[:,i] = signal.filtfilt(b, a, data[:,i])
	filtered = filtered + (data[0] - filtered[0])
	f0 = notch
	q = 30
	b, a = signal.iirnotch(f0, q, fs)
	notched = np.zeros(data.shape)
	for i in np.arange(0, number_of_channels):
		notched[:, i] = signal.filtfilt(b, a, filtered[:, i])
	notched_df = pd.DataFrame(notched, columns=column_names)
	return notched_df


# ------------------
# DATASET OUTLINE
# ------------------
def create_timestamps(data):
    timestamps = []
    for i in range(0, data.shape[0]):
        timestamps.append(START_TIME_ICTAL + (i * FS))
    return timestamps


# ------------------
# PROCESS DATA
# ------------------
def create_merge_dataset(data, split_point, start_time_interictal, 
                         end_time_interictal, start_time_ictal, end_time_ictal):
    dataset = []
    dataset_targets = []
    labels = pd.read_csv(PATH_LABELS, header = None)
    
    for column in data:
        
        print("Reading data for electrode " + column)
        
        col_list = data[column].tolist()
        col_data = labels.loc[labels[0] == column]
        
        col_start_time = col_data.iat[0, 1]
        col_end_time = col_data.iat[0, 2]
                
        if(col_start_time == '-' or col_end_time == '-'):
            col_start_time = int(start_time_ictal + 1)
            col_end_time = int(start_time_ictal + 1)
        else:
            col_start_time = int(col_start_time)
            col_end_time = int(col_end_time)
            
        # first process interictal data
        for index in range(SEQUENCE_LEN, split_point, STEP_SIZE):
            sequence = col_list[(index - SEQUENCE_LEN) : index]
            sequence = [[i] for i in sequence]
            dataset.append(sequence)
    
            sequence_end_time = start_time_interictal + (index * FS * DOWN_SAMPLE_FACTOR)
            
            if(sequence_end_time >= col_start_time and 
               sequence_end_time <= col_end_time):
                dataset_targets.append(1)
            else:
                dataset_targets.append(0)
        
        # then process ictal data
        for index in range(SEQUENCE_LEN, data.shape[0] - split_point, STEP_SIZE):
            new_index = index + split_point
            sequence = col_list[(new_index - SEQUENCE_LEN) : new_index]
            sequence = [[i] for i in sequence]
            dataset.append(sequence)
    
            sequence_end_time = start_time_ictal + (index * FS * DOWN_SAMPLE_FACTOR)
            
            # print(sequence_end_time, col_start_time)
            if(sequence_end_time >= col_start_time and 
               sequence_end_time <= col_end_time):
                dataset_targets.append(1)
            else:
                dataset_targets.append(0)
                
    dataset = np.array(dataset)
    dataset_targets = np.array(dataset_targets)
    
    return dataset, dataset_targets

# DEPRECATED: only for single, continous start and end time
def create_dataset(data, start_time, end_time):
    dataset = []
    dataset_targets = []
    labels = pd.read_csv(PATH_LABELS, header = None)
    
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


# ------------------
# TEST MODEL
# ------------------
def model_acc(model_name, test, targets):
    pickle_name = model_name + '.pkl'
    model = load_model(pickle_name)
    preds = model.predict_classes(test)
    acc = accuracy_score(targets, preds)
    cm = confusion_matrix(targets, preds)
    scores = precision_recall_fscore_support(targets, preds, average = 'macro')
    
    print("Number of seizing instances in targets: " + 
          str(np.count_nonzero(targets == 1)))
    print("Number of non-seizing instances in targets: " + 
          str(np.count_nonzero(targets == 0)))
    
    print("Number of seizing instances in predictions: " + 
          str(np.count_nonzero(preds == 1)))
    print("Number of non-seizing instances in predictions: " + 
          str(np.count_nonzero(preds == 0)))
    
    return acc, cm, scores


# ------------------
# MAIN METHOD
# ------------------
if __name__=="__main__":
        
    # --------------------
    # DATA WRANGLING
    # --------------------
    # interictal
    data = get_data(PATH_INTERICTAL)
    timestamps = create_timestamps(data)
    data_filtered = iEEG_data_filter(data, FS, 0.16, 200, 60)
    fs_downSample = FS / DOWN_SAMPLE_FACTOR
    data_filtered_tmp = signal.decimate(data_filtered, DOWN_SAMPLE_FACTOR, axis = 0)
    data_filtered = pd.DataFrame(data_filtered_tmp, columns = data_filtered.columns); del data_filtered_tmp
    data_interictal = data_filtered
    
    # ictal
    data = get_data(PATH_ICTAL)
    timestamps = create_timestamps(data)
    data_filtered = iEEG_data_filter(data, FS, 0.16, 200, 60)
    fs_downSample = FS / DOWN_SAMPLE_FACTOR
    data_filtered_tmp = signal.decimate(data_filtered, DOWN_SAMPLE_FACTOR, axis = 0)
    data_filtered = pd.DataFrame(data_filtered_tmp, columns = data_filtered.columns); del data_filtered_tmp
    data_ictal = data_filtered
    
    # concatenate
    data = pd.concat([data_interictal, data_ictal], ignore_index = True)
    dataset, dataset_targets = create_merge_dataset(data, data_interictal.shape[0],
                                              START_TIME_INTERICTAL, END_TIME_INTERICTAL,
                                              START_TIME_ICTAL, END_TIME_ICTAL)
    
    # --------------------
    # TESTING MODEL
    # --------------------
    cnn_acc, cnn_cm, cnn_scores = model_acc(PATH_MODEL, dataset, dataset_targets)
    print("CNN WaveNet Test Set Accuracy: ")
    print("%.4f" % round(cnn_acc, 4))   
    print("CNN WaveNet Test Set Confusion Matrix: ")
    print(cnn_cm)
    