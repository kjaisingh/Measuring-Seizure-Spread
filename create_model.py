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
EPOCHS = 1
BS = 128
LR = 0.001
PATH = "../datasets/hup138.pickle"
FS = 1024


# ------------------
# LOAD DATA
# ------------------
def get_data(path):
    with open(path, 'rb') as f: data, fs = pickle.load(f)
    
    labels = pd.read_csv("hup138-labels.csv", header = None)
    labels_list = labels[0].tolist()
    data = data[data.columns.intersection(labels_list)]
    return data


# ------------------
# DATASET OUTLINE
# ------------------
def create_timestamps(data):
    timestamps = []
    for i in range(0, data.shape[0]):
        timestamps.append(START_TIME + (i * FS))
    return timestamps
    



# ------------------
# PROCESS DATA
# ------------------
def create_dataset(data):
    
    labels = pd.read_csv("hup138-labels.csv", header = None)
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
    
            sequence_end_time = START_TIME + (index * FS)
            
            if(sequence_end_time >= col_end_time):
                dataset_targets.append(1)
            else:
                dataset_targets.append(0)
        
    return dataset, dataset_targets


# ------------------
# CREATE MODELS
# ------------------
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(256, input_shape = (SEQUENCE_LEN, 1)))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    return model

def create_cnn_model():
    model = Sequential()
    model.add(LSTM(256, input_shape = (SEQUENCE_LEN, 1)))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    return model


# ------------------
# TRAIN MODELS
# ------------------
def train_lstm_model(model, model_name):
    adam = Adam(lr = LR)
    
    chk = ModelCheckpoint((model_name + '.pkl'), 
                          monitor = 'val_accuracy', 
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
    
    return model


# ------------------
# TEST MODEL
# ------------------
def model_acc(model_name):
    pickle_name = model_name + '.pkl'
    model = load_model(pickle_name)
    y_preds = model.predict_classes(x_test)
    acc = accuracy_score(y_test, y_preds)
    return acc


# ------------------
# MAIN METHOD
# ------------------
if __name__=="__main__":
    
    # get and create dataset
    data = get_data(PATH)
    timestamps = create_timestamps(data)
    
    # dataset dimensions should be (# samples, 2048, 1)
    # target dataset dimensions should be (# samples)
    # 0 indicates normal activity, 1 indicates seizing
    dataset = []
    dataset_targets = []
    dataset, dataset_targets = create_dataset(data)
    
    # split dataset into train, test and validation
    dataset = np.array(dataset)
    dataset_targets = np.array(dataset_targets)
    
    x_train, x_test, y_train, y_test = train_test_split(dataset, 
                                                        dataset_targets, 
                                                        test_size = 0.2)
    x_test, x_val, y_test, y_val = train_test_split(x_test, 
                                                    y_test, 
                                                    test_size = 0.5)
    
    # LSTM
    lstm_model = create_lstm_model()
    lstm_model_name = 'lstm-eeg-model'
    lstm_model = train_lstm_model(lstm_model, lstm_model_name)
    
    # CNN
    cnn_model = create_lstm_model()
    cnn_model_name = 'cnn-eeg-model'
    cnn_model = train_lstm_model(cnn_model, cnn_model_name)
    
    # test models
    lstm_acc = model_acc(lstm_model_name)
    cnn_acc = model_acc(cnn_model_name)
    print("Final LSTM Accuracy: " + lstm_acc)
    print("Final CNN Accuracy: " + cnn_acc)
