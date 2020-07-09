# Measuring Seizure Spread

### get_seizure_data.py
* Script to get seizure data from a particular iEEG dataset and save it as a pickle.
* Sample execution: python -c 'import get_seizure_data; get_seizure_data. get_seizure_data("karanjaisingh", "<password>", "HUP138_phaseII", 415839606029, 416311906098, "/Users/jaisi8631/Desktop/Davis Lab/datasets/hup138.pickle")'

### create_model.py
* Process training data for model and train LSTM model.
* Currently utilizes a baseline model.
