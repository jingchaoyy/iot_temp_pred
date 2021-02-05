# Hourly Temperature Prediction in Python


## Preparing your dataset:

Dataset is large, please contact the author Jingchao Yang (jyang43@gmu.edu) for direct access

Place the dataset under Data folder in the same directory before running the code 

## Requirements:
- Python 3.7
- PyTorch 1.7.0
- Pandas 1.0.1
- scikit-learn
- scipy
- numpy
- matplotlib

## Category of models:

* [multistep_lstm](multistep_lstm) indludes python files for LSTM model building and training. 
* [multistep_others](multistep_others) includes comparison model ARIMA and XGBoost.


### - LSTM:

[Tutorial](https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/) for LSTM using pytorch 

To run our LSTM model, go to the [directory](multistep_lstm) and using the command: 

python run_auto.py --transLearn=False

Argument "transLearn" controls regular model training (set to False) or transfer learning (set to True), more detail about transfer learning will be published in the next coming paper.

### - ARIMA:

[Tutorial](https://www.kaggle.com/sumi25/understand-arima-and-tune-p-d-q) for ARIMA 

### - XGBoost:

[Tutorial](https://www.kaggle.com/furiousx7/xgboost-time-series) for XGBoost 
