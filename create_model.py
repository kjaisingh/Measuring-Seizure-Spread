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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ------------------
# LSTM IMPORTS
# ------------------
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


# ------------------
# CNN IMPORTS
# ------------------
from keras.optimizers import SGD
from keras.models import Model 
from keras.layers import Reshape
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping


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
NUM_CLASSES = 2


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
# LSTM MODEL
# ------------------
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(256, input_shape = (SEQUENCE_LEN, 1)))
    model.add(Dense(1, activation = 'sigmoid'))
    print(model.summary())
    return model

def train_lstm_model(model, model_name, x_train, y_train, x_val, y_val):
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
# CNN MODEL
# ------------------
def create_cnn_model():
    model = Sequential()
    input_shape = (SEQUENCE_LEN * 1)
    model.add(Reshape((SEQUENCE_LEN, 1), input_shape = (input_shape,)))
    model.add(Conv1D(1000, 100, activation='relu', input_shape = (SEQUENCE_LEN, 1)))
    model.add(Conv1D(500, 50, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(Conv1D(10, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    print(model.summary())
    return model

def train_cnn_model(model, model_name, x_train, y_train, x_val, y_val):
    adam = Adam(lr = LR)
    chk = ModelCheckpoint((model_name + '.pkl'), 
                          monitor = 'val_acc', 
                          save_best_only = True,
                          mode= 'max',
                          verbose = 1)
    callbacks_list = [
        chk,
        EarlyStopping(monitor='acc', patience=1)
    ]
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer= adam, 
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size = BS,
              epochs = EPOCHS,
              callbacks = callbacks_list,
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
    lstm_model = train_lstm_model(lstm_model, lstm_model_name, 
                                  x_train, y_train, x_val, y_val)
    
    # CNN
    cnn_model = create_lstm_model()
    cnn_model_name = 'cnn-eeg-model'
    cnn_model = train_lstm_model(cnn_model, cnn_model_name, 
                                 x_train, y_train, x_val, y_val)
    
    # test models
    lstm_acc = model_acc(lstm_model_name)
    print("LSTM Test Set Accuracy: ")
    print("%.4f" % round(lstm_acc, 4))
    
    print("\n-------------------------------\n")
    
    cnn_acc = model_acc(cnn_model_name)
    print("CNN Test Set Accuracy: ")
    print("%.4f" % round(cnn_acc, 4))
