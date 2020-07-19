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
import math
from scipy import signal
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV

import keras
import tensorflow as tf
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
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D


# ------------------
# CONSTANTS
# ------------------
START_TIME = 415839606029
END_TIME = 416311906098
FS = 1024

STEP_SIZE = 4096
SEQUENCE_LEN = 2048
SEQUENCE_PCA = 1000

EPOCHS = 1
BS = 32
LR = 0.001
NUM_CLASSES = 2
MOMENTUM = 0.1
DECAY = 1e-6

GS_EPOCHS = [1, 2, 3]
GS_BS = [16, 32, 64]
GS_OPTIMIZERS = ['adam', 'rmsprop']

ADAM_DEFAULT = 'adam'
SGD_DEFAULT = 'sgd'
ADAM_CUSTOM = Adam(lr = LR)
SGD_CUSTOM = SGD(lr = LR, momentum = MOMENTUM, decay = DECAY, nesterov = True)

PATH = "../datasets/hup138.pickle"


# ------------------
# LOSS HISTORY
# ------------------
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'acc':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('acc'))


# ------------------
# LOAD DATA
# ------------------
def get_data(path):
    with open(path, 'rb') as f: data, fs = pickle.load(f)
    
    labels = pd.read_csv("hup138-labels.csv", header = None)
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
        timestamps.append(START_TIME + (i * FS))
    return timestamps


# ------------------
# PROCESS DATA
# ------------------
def create_dataset(data):
    dataset = []
    dataset_targets = []
    
    labels = pd.read_csv("hup138-labels.csv", header = None)
    for column in data:
        
        print("Currently on Electrode: " + column)
        
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
                
    dataset = np.array(dataset)
    dataset_targets = np.array(dataset_targets)
    
    return dataset, dataset_targets

def scale_data(dataset):
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset.reshape(-1, dataset.shape[-1])).reshape(dataset.shape)
    return dataset

def create_class_weights(y_train):
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes = np.unique(y_train),
                                                      y = y_train)
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict

def apply_pca(dataset, components):
    pca = decomposition.PCA(n_components = components)
    pca.fit(components)
    dataset = pca.transform(dataset)
    return dataset


# ------------------
# LSTM MODEL
# ------------------
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(256, input_shape = (SEQUENCE_LEN, 1), dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))
    model.add(LSTM(32, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))
    model.add(LSTM(32, return_sequences = False))
    model.add(Dense(1, activation = 'sigmoid'))
    print(model.summary())
    return model

def train_lstm_model(model, model_name, x_train, y_train, x_val, y_val):    
    chk = ModelCheckpoint((model_name + '.pkl'), 
                          monitor = 'val_accuracy', 
                          save_best_only = True, 
                          mode = 'max', 
                          verbose = 1)
    lstm_history = LossHistory()
    callbacks_list = [
        chk,
        lstm_history,
        EarlyStopping(monitor = 'acc', patience = 1)
    ]
    
    model.compile(loss = 'binary_crossentropy', 
                  optimizer = ADAM_CUSTOM, 
                  metrics = ['accuracy'])
    
    class_weights = create_class_weights(y_train)
    
    model.fit(x_train, y_train, 
              class_weight = class_weights,
              epochs = EPOCHS, 
              batch_size = BS, 
              callbacks = callbacks_list, 
              validation_data = (x_val, y_val))
    
    plot_batch_losses(lstm_history, 'lstm-history')
    return model

def lstm_gridsearch(optimizer = 'adam'):
    model = Sequential()
    model.add(LSTM(256, input_shape = (SEQUENCE_LEN, 1), dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))
    model.add(LSTM(32, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))
    model.add(LSTM(32, return_sequences = False))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model    


# ------------------
# CNN MODEL
# ------------------
def create_cnn_model():
    model = Sequential()
    model.add(Conv1D(500, 100, activation='relu', input_shape = (SEQUENCE_LEN, 1)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(10, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    print(model.summary())
    return model

def train_cnn_model(model, model_name, x_train, y_train, x_val, y_val):
    chk = ModelCheckpoint((model_name + '.pkl'), 
                          monitor = 'val_acc', 
                          save_best_only = True,
                          mode = 'max',
                          verbose = 1)
    cnn_history = LossHistory()
    callbacks_list = [
        chk,
        cnn_history,
        EarlyStopping(monitor = 'acc', patience = 1)
    ]
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = ADAM_CUSTOM, 
                  metrics = ['accuracy'])
    
    class_weights = create_class_weights(y_train)

    model.fit(x_train, y_train,
              class_weight = class_weights,
              batch_size = BS,
              epochs = EPOCHS,
              callbacks = callbacks_list,
              validation_data = (x_val, y_val))
    
    plot_batch_losses(cnn_history, 'cnn-history')
    return model

def cnn_gridsearch(optimizer = 'adam'):
    model = Sequential()
    model.add(Conv1D(500, 100, activation='relu', input_shape = (SEQUENCE_LEN, 1)))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(100, 10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv1D(10, 10, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
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

def plot_batch_losses(history, plot_name):
    y1 = history.history['loss']
    y2 = history.history['acc']
    x1 = np.arange(len(y1))
    k = len(y1) / len(y2)
    x2 = np.arange(k, len(y1) + 1, k)
    fig, ax = plt.subplots()
    line1, = ax.plot(x1, y1, label = 'loss')
    line2, = ax.plot(x2, y2, label = 'acc')
    plt.savefig(plot_name + '.png')
    plt.show()

def build_gridsearch(build_function, x_train, y_train, model_name):
    grid = {'epochs': GS_EPOCHS,
            'batch_size': GS_BS,
            'optimizer': GS_OPTIMIZERS}
    load_model = KerasClassifier(build_fn = build_function)
    
    # train using GridSearch 
    validator = GridSearchCV(load_model,
                             param_grid = grid,
                             scoring = 'accuracy',
                             n_jobs = 1)
    validator.fit(x_train, y_train)
    
    # show results
    print('Grid Search: ')
    summary = validator.score_summary()
    print(summary)
    
    # retain best model
    print('Best Parameters: ')
    model = validator.best_estimator_.model
    print(validator.best_params_)
    model.save(model_name + '.pkl')
    
    return model


# ------------------
# MAIN METHOD
# ------------------
if __name__=="__main__":
    
    # get and create dataset
    data = get_data(PATH)
    timestamps = create_timestamps(data)

    # filter and downsample data
    data_filtered = iEEG_data_filter(data, FS, 0.16, 200, 60)
    down_sample_factor = 10
    fs_downSample = FS / down_sample_factor
    data_filtered_tmp = signal.decimate(data_filtered, down_sample_factor, axis=0)
    data_filtered = pd.DataFrame(data_filtered_tmp, columns = data_filtered.columns); del data_filtered_tmp
    data = data_filtered

    # dataset dimensions should be (# samples, 2048, 1)
    # target dataset dimensions should be (# samples)
    # 0 indicates normal activity, 1 indicates seizing
    dataset, dataset_targets = create_dataset(data)
    
    """
    OPTIONALPREPROCESSING METHODS: 
    dataset = scale_data(datset)
    dataset = apply_pca(dataset, SEQUENCE_PCA)
    """ 
    
    # split dataset into train, test and validation
    x_train, x_test, y_train, y_test = train_test_split(dataset, 
                                                        dataset_targets, 
                                                        test_size = 0.2)
    x_test, x_val, y_test, y_val = train_test_split(x_test, 
                                                    y_test, 
                                                    test_size = 0.5)
    
    # --------------------
    # OPTION 1: Regular
    # --------------------
    # train LSTM
    lstm_model = create_lstm_model()
    lstm_model_name = 'eeg-model-lstm'
    lstm_model = train_lstm_model(lstm_model, lstm_model_name, 
                                  x_train, y_train, x_val, y_val)
    
    # test LSTM
    lstm_acc = model_acc(lstm_model_name)
    print("LSTM Test Set Accuracy: ")
    print("%.4f" % round(lstm_acc, 4))
    
    """
    # train CNN
    cnn_model = create_cnn_model()
    cnn_model_name = 'eeg-model-cnn'
    cnn_model = train_cnn_model(cnn_model, cnn_model_name, 
                                 x_train, y_train, x_val, y_val)
    
    # test CNN
    cnn_acc = model_acc(cnn_model_name)
    print("CNN Test Set Accuracy: ")
    print("%.4f" % round(cnn_acc, 4))
    
    # plot model histories
    
    
    
    # --------------------
    # OPTION 2: GridSearch
    # --------------------
    # train and test LSTM
    gs_lstm_name = 'egg-lstm-gridsearch'
    gs_lstm = build_gridsearch(lstm_gridsearch, x_train, y_train, gs_lstm_name)
    gs_lstm_acc = model_acc(gs_lstm_name)
    print("LSTM Test Set Accuracy: ")
    print("%.4f" % round(gs_lstm_acc, 4))
    
    # train and test CNN
    gs_cnn_name = 'eeg-cnn-gridsearch'
    gs_cnn = build_gridsearch(cnn_gridsearch, x_train, y_train, gs_cnn_name)
    gs_cnn_acc = model_acc(gs_cnn_name)
    print("CNN Test Set Accuracy: ")
    print("%.4f" % round(gs_cnn_acc, 4))
    """
    

    
    
