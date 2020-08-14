# Measuring Seizure Spread

### File: get_seizure_data.py
* Script to get seizure data from a particular iEEG dataset and save it as a pickle.
* Sample execution: python -c 'import get_seizure_data; get_seizure_data. get_seizure_data("karanjaisingh", "INSERT_PASSWORD_HERE", "HUP138_phaseII", 415839606029, 416311906098, "/Users/jaisi8631/Desktop/Davis Lab/datasets/hup138.pickle")'

### File: create_model.py
* Process training data to create dataset which can be used to train models.
* Trains and tests a baseline LSTM model.
* Trains and tests a baseline CNN model.

### File: create_predictions.py
* Requires dataset to be downloaded from iEEG and labels csv file to be created.
* Predicts the state of EEG sequences as 0 or 1, and outputs summaries of the model's overall accuracy.
