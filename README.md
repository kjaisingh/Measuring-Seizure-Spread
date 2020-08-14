# Measuring Seizure Spread

### File: get_seizure_data.py
* Script to get seizure data from a particular iEEG dataset and save it as a pickle.
* Sample execution: python -c 'import get_seizure_data; get_seizure_data. get_seizure_data("karanjaisingh", "INSERT_PASSWORD_HERE", "HUP138_phaseII", 415839606029, 416311906098, "/Users/jaisi8631/Desktop/Davis Lab/datasets/hup138.pickle")'

### File: create_model.py
* Process training data to create dataset which can be used to train models.
* Trains and tests a baseline LSTM model.
* Trains and tests a baseline CNN model.