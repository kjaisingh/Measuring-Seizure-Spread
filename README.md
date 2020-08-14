# Measuring Seizure Spread
Repository to hold code relevant to the paper titled 'Quantifying Seizure Spread' by Andy Revell.

### get_seizure_data.py
* Script to get seizure data from a particular iEEG dataset and save it as a pickle.
* Sample execution: 
```
python -c 'import get_seizure_data; get_seizure_data. get_seizure_data("karanjaisingh", "INSERT_PASSWORD_HERE", "HUP138_phaseII", 415839606029, 416311906098, "/Users/jaisi8631/Desktop/Davis Lab/datasets/hup138.pickle")'
```

### create_model.py
* Process training data to create dataset which can be used to train models.
* Trains and tests a baseline LSTM model, a baseline CNN model and a WaveNet CNN model.
* Sample execution:
```
python create_model.py
```

### create_predictions.py
* Requires datasets to have been downloaded from iEEG and labels csv file to have been created.
* Predicts the state of EEG sequences as 0 or 1, and outputs a summary of the model's accuracy on the dataset.
* 'Notes' heading has details of how the structure of data should be organised before execution.
* Sample execution: 
```
python create_predictions.py --start_interictal 356850680000 --end_interictal 356903099171 --start_ictal 402704260829 --end_ictal 402756680000 --dataset_id hup172 --model_name eeg-model-cnn-wavenet
```
* Condensed execution: 
```
python create_predictions.py -a 356850680000 -b 356903099171 -c 402704260829 -d 402756680000 -i hup172 -m eeg-model-cnn-wavenet
```
