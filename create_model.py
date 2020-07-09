#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:58:18 2020

@author: jaisi8631
"""

# ------------------
# REGULAR IMPORTS
# ------------------
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ------------------
# KERAS IMPORTS
# ------------------
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score


# ------------------
# CONSTANTS
# ------------------
START_TIME = 415839606029
END_TIME = 416311906098
SEQUENCE_LEN = 2048
STEP_SIZE = 4096
EPOCHS = 10
BS = 64


# ------------------
# LOAD DATA
# ------------------
with open("../datasets/hup138.pickle", 'rb') as f: data, fs = pickle.load(f)

labels = pd.read_csv("hup138-labels.csv", header = None)
labels_list = labels[0].tolist()
data = data[data.columns.intersection(labels_list)]


# ------------------
# DATASET OUTLINE
# ------------------
timestamps = []
for i in range(0, data.shape[0]):
    timestamps.append(START_TIME + (i * fs))
    
# dataset dimensions should be (# samples, 2048, 1)
dataset = []

# target dataset dimensions should be (# samples)
# 0 indicates normal activity, 1 indicates seizing
dataset_targets = []


# ------------------
# PROCESS DATA
# ------------------
for column in data:
    
    print("Currently on Column: " + column)
    
    col_list = data[column].tolist()
    col_data = labels.loc[labels[0] == column]
    
    col_start_time = col_data.iat[0, 1]
    col_end_time = col_data.iat[0, 2]
    
    if(col_start_time == '-' or col_end_time == '-'):
        col_start_time = int(START_TIME + 1)
        col_end_time = int(START_TIME + 1)
    else:
        col_start_time = int(col_start_time)
        col_end_time = int(col_end_time)  
    
    # index indicates the ending row number of the sequence
    for index in range(SEQUENCE_LEN, data.shape[0], STEP_SIZE):
        sequence = col_list[(index - SEQUENCE_LEN):index]
        sequence = [[i] for i in sequence]
        dataset.append(sequence)

        sequence_end_time = START_TIME + (index * fs)
        
        if(sequence_end_time >= col_end_time):
            dataset_targets.append(1)
        else:
            dataset_targets.append(0)

    
# ------------------
# SPLIT DATA
# ------------------
dataset = np.array(dataset)
dataset_targets = np.array(dataset_targets)

x_train, x_test, y_train, y_test = train_test_split(dataset, 
                                                    dataset_targets, 
                                                    test_size = 0.2)
x_test, x_val, y_test, y_val = train_test_split(x_test, 
                                                y_test, 
                                                test_size = 0.5)


# ------------------
# CREATE MODEL
# ------------------
model = Sequential()
model.add(LSTM(256, input_shape = (SEQUENCE_LEN, 1)))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# ------------------
# TRAIN MODEL
# ------------------
adam = Adam(lr=0.001)

chk = ModelCheckpoint('baseline.pkl', 
                      monitor = 'val_acc', 
                      save_best_only = True, 
                      mode = 'max', 
                      verbose = 1)

model.compile(loss = 'binary_crossentropy', 
              optimizer = adam, 
              metrics = ['accuracy'])

model.fit(x_train, y_train, 
          epochs = EPOCHS, 
          batch_size = BS, 
          callbacks = [chk], 
          validation_data = (x_val, y_val))


# ------------------
# VALIDATE MODEL
# ------------------
model = load_model('baseline.pkl')
y_preds = model.predict_classes(x_test)
acc = accuracy_score(y_test, y_preds)
