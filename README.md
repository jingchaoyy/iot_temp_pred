# Hourly Temperature Prediction in Python

* This project has been validated [here](https://github.com/stccenter/IoT-based-Temperature-Prediction) with a detailed tutorial for experiment duplication.
* Check out the slides [AAG 2021 Presentation](AAG_2021_IoTbased_Fine-scale_Urban_Temperature_Forecasting_Jingchao.pdf) for more project details.

# publication
Yang, J., Yu, M., Liu, Q., Li, Y., Duffy, D. Q., & Yang, C. (2022). A high spatiotemporal resolution framework for urban temperature prediction using IoT data. Computers & Geosciences, 159, 104991.

## Preparing your dataset:

Dataset can be downloaded [here](https://exchangelabsgmu-my.sharepoint.com/:f:/g/personal/jyang43_masonlive_gmu_edu/En-TZLF4UVBAqyCtiyQOYM0BU3leFL4TSCJd18xoIXovGA?e=b3LTcq). Please contact the author Jingchao Yang (jyang43@gmu.edu) for direct access if link expires.

* Place the dataset in the data folder to avoid additional path setup before running the code 

**Note**: All data has been preprocessed to csv format, raw data can be accessed from [weather underground](https://www.wunderground.com/) and [GeoTab](https://data.geotab.com/weather/temperature). Toolset for preprocessing raw data can be accessed upon request.

## Requirements:
- Python 3.7
- PyTorch 1.7.0 (code has GPU support, but can run without) 
- Pandas 1.0.1
- scikit-learn
- scipy
- numpy
- matplotlib
- tqdm
- pmdarima
- xgboost

## Category of models:

* [multistep_lstm](multistep_lstm) indludes python files for LSTM model building and training. 
* [multistep_others](multistep_others) includes comparison model ARIMA and XGBoost.


### - LSTM:

To run our LSTM model for regional training, go to the [directory](multistep_lstm) and use the command

```python run_auto.py```

LSTM was also developed to support transfer learning with command

```python run_auto.py --transLearn```

**Note**: Model training can take much longer time without GPU support. LA Dataset already includes trained models and ready for transfer learning, user can delete the content inside the LA/output to retrain

Model output will be store in the data/output folder

### - Other models 

Creat result folder under [multistep_others](multistep_others) for model output. ARIMA and XGBoost are for model comparison and were not developed for transfer learning 

#### - ARIMA:

To use our ARIMA model, go to [multistep_others](multistep_others) and use the command

```python auto_arima_run.py```

#### - XGBoost:

To use our XGBoost model, go to [multistep_others](multistep_others) and use the command

```python xgboost_run.py```


## Useful links
* [Tutorial](https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/) for LSTM using pytorch 
* [Tutorial](https://www.kaggle.com/sumi25/understand-arima-and-tune-p-d-q) for ARIMA 
* [Tutorial](https://www.kaggle.com/furiousx7/xgboost-time-series) for XGBoost 
