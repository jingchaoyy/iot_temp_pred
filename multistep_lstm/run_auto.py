"""
Created on  10/2/2020
@author: Jingchao Yang

auto model training and cross region testing with predefined input/prediction combination
"""
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
from sklearn.metrics import mean_squared_error
from statistics import mean
import torch
from torch.utils.data import TensorDataset, DataLoader
from multistep_lstm import multistep_lstm_pytorch
from multistep_lstm import model_train
# import multistep_lstm_pytorch as multistep_lstm_pytorch
# import model_train as model_train
from sklearn import preprocessing
import numpy as np
from numpy import isnan
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import os
import glob


'''pytorch'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
multi_variate_mode = True

'''data'''
# self-training and test path
load_model = False
out_path = r'..\data\LA\output'
exp_path = r'..\data\LA'
iot_path = exp_path + r'\IoT'
wu_path = exp_path + r'\WU'

# # trans mode test path
# load_model = True
# model_load_path = r'..\data\LA\output'
# out_path = r'..\data\LA\output\trans_model_test\Chicago_tuning'
# exp_path = r'..\data\Chicago'
# iot_path = exp_path + r'\IoT'
# wu_path = exp_path + r'\WU'


geohash_df = pd.read_csv(iot_path + r'\nodes_missing_5percent.csv',
                         usecols=['Geohash'])
iot_sensors = geohash_df.values.reshape(-1)
iot_df = pd.read_csv(iot_path + r'\preInt_matrix_full.csv',
                     usecols=['datetime'] + iot_sensors.tolist(), index_col=['datetime'])

# ext_name = ['humidity', 'windSpeed']
ext_name = ['humidity', 'windSpeed', 'dewPoint', 'precipProbability', 'pressure', 'cloudCover', 'uvIndex']
ext_data_scaled = []
if multi_variate_mode:
    ext_data_path = wu_path + r'\byAttributes'
    for ext in ext_name:
        ext_df = pd.read_csv(ext_data_path + f'\{ext}.csv', index_col=['datetime'])
        while ext_df.isnull().values.any():
            model_train.fill_missing(ext_df.values)
        print(f'NaN value in {ext} df?', ext_df.isnull().values.any())
        ext_data_scaled.append(model_train.min_max_scaler(ext_df))
    iot_wu_match_df = pd.read_csv(exp_path + r'\iot_wu_colocate.csv', index_col=0)

while iot_df.isnull().values.any():
    model_train.fill_missing(iot_df.values)
print('NaN value in IoT df?', iot_df.isnull().values.any())

'''all stations data preprocessing'''
selected_vars = iot_sensors
dataset = iot_df

print('selected sensors', dataset.columns)

dataset = dataset.values
dataset[dataset < 0] = 0
print('size', dataset.shape)

# find max and min values for normalization
norm_min = dataset.min()
norm_max = dataset.max()
print('dataset min, max', norm_min, norm_max)

# normalize the data
dataset = (dataset - norm_min) / (norm_max - norm_min)
print('normalized dataset min, max', dataset.min(), dataset.max())


'''start experiments'''
experiments = [(24, 1), (24, 4), (24, 8), (24, 12), (36, 12), (48, 12), (72, 12), (72, 24), (72, 36), (72, 48),
               (120, 48), (144, 48), (120, 72), (144, 72), (168, 120)]

for exp in range(len(experiments)):
    train_window, output_size = experiments[exp]
    print('#################current exp', train_window, output_size)

    output_path = out_path + f'\\{str(train_window)}_{str(output_size)}'
    try:
        # Create target Directory
        os.mkdir(output_path)
        print("Directory ", output_path, " Created ")
    except FileExistsError:
        print("Directory ", output_path, " already exists")

    # separate train and test stations
    train_stations = set(np.random.choice(selected_vars, int(len(selected_vars) * 0.7), replace=False))
    test_stations = set(selected_vars) - train_stations

    train_data_raw = iot_df[train_stations]
    test_data_raw = iot_df[test_stations]

    print(train_data_raw.shape)
    print(test_data_raw.shape)
    print(train_data_raw.columns)

    if not multi_variate_mode:
        train_data = multistep_lstm_pytorch.Dataset(train_data_raw,
                                                    (norm_min, norm_max),
                                                    train_window, output_size)
        test_data = multistep_lstm_pytorch.Dataset(test_data_raw,
                                                   (norm_min, norm_max),
                                                   train_window,
                                                   output_size,
                                                   test_station=True)
    else:
        train_data = multistep_lstm_pytorch.Dataset_multivariate(train_data_raw,
                                                                 (norm_min, norm_max),
                                                                 train_window,
                                                                 output_size,
                                                                 ext_data_scaled,
                                                                 ext_name,
                                                                 iot_wu_match_df)
        test_data = multistep_lstm_pytorch.Dataset_multivariate(test_data_raw,
                                                                (norm_min, norm_max),
                                                                train_window,
                                                                output_size,
                                                                ext_data_scaled,
                                                                ext_name,
                                                                iot_wu_match_df,
                                                                test_station=True)

    print('Number of stations in training data: ', len(train_data))
    print('Number of stations in testing data: ', len(test_data))

    print("Training input and output for each station: %s, %s" % (train_data[0][0].shape, train_data[0][1].shape))
    print("Validation input and output for each station: %s, %s" % (train_data[0][2].shape, train_data[0][3].shape))
    print("Testing input and output for each station: %s, %s" % (test_data[0][0].shape, test_data[0][1].shape))

    '''initialize the model'''
    num_epochs = 15
    epoch_interval = 1
    hidden_size = 12
    loss_func, model, optimizer = multistep_lstm_pytorch.initial_model(input_size=train_data[0][0].shape[-1],
                                                                       hidden_size=hidden_size,
                                                                       output_size=output_size,
                                                                       learning_rate=0.001
                                                                       )
    if not load_model:
        model = model_train.start_training(model, train_data, device, loss_func, optimizer, num_epochs)

        # save trained model
        modelName = int(time.time())
        torch.save(model.state_dict(), output_path + f'\\input{train_window}_pred{output_size}_{modelName}.pt')
        print('model saved')
    else:
        model_path = glob.glob(model_load_path + f'\\{str(train_window)}_{str(output_size)}' + r'\*.pt')
        model.load_state_dict(torch.load(model_path[0], map_location=device))
        model.eval()
        model = model_train.start_training(model, train_data, device, loss_func, optimizer, num_epochs)

    # Predict the training dataset of training stations and testing dataset of testing stations
    train_pred_orig_dict = dict()
    for idx in range(len(train_data)):
        station = train_data.keys[idx]
        with torch.no_grad():
            train_pred = model(train_data[idx][0][:, :, 0, :].to(device))
            train_pred_trans = train_pred * (norm_max - norm_min) + norm_min

            train_orig = train_data[idx][1][:, :, 0, :].reshape(train_pred.shape).to(device)
            train_orig_trans = train_orig * (norm_max - norm_min) + norm_min

            train_pred_orig_dict[station] = (train_pred_trans, train_orig_trans)

    test_pred_orig_dict = dict()
    for idx in range(len(test_data)):
        station = test_data.keys[idx]
        with torch.no_grad():
            test_pred = model(test_data[idx][0][:, :, 0, :].to(device))
            test_pred_trans = test_pred * (norm_max - norm_min) + norm_min

            test_orig = test_data[idx][1][:, :, 0, :].reshape(test_pred.shape).to(device)
            test_orig_trans = test_orig * (norm_max - norm_min) + norm_min

            test_pred_orig_dict[station] = (test_pred_trans, test_orig_trans)

    print(list(test_pred_orig_dict.keys())[0])

    # plot baseline and predictions
    d = {'ori': test_pred_orig_dict[list(test_pred_orig_dict.keys())[0]][1][:, 0].data.tolist(),
         'pred': test_pred_orig_dict[list(test_pred_orig_dict.keys())[0]][0][:, 0].data.tolist()}
    pred_df = pd.DataFrame(data=d)
    pred_df.to_csv(output_path + r'\pred.csv')
    # pred_df.plot()
    # plt.xlabel('time (hour)')
    # plt.ylabel('temperature (F)')
    # plt.show()

    # getting r2 score for mode evaluation
    model_score = r2_score(pred_df.pred, pred_df.ori)
    print("R^2: ", model_score)

    # calculate root mean squared error
    rmse_by_station_train, mae_by_station_train, rmse_by_hour_train, mae_by_hour_train = model_train.result_evaluation(
        train_pred_orig_dict)
    rmse_by_station_test, mae_by_station_test, rmse_by_hour_test, mae_by_hour_test = model_train.result_evaluation(
        test_pred_orig_dict)

    '''RMSE'''
    rmse_by_station_train.to_csv(output_path + r'\trainScores_C.csv')
    np.savetxt(output_path + r'\trainScores_C_by_hour.csv', rmse_by_hour_train, delimiter=",")

    print('max test RMSE', round(rmse_by_station_test.max(), 2))
    print('min test RMSE', round(rmse_by_station_test.min(), 2))
    rmse_by_station_test.to_csv(output_path + r'\testScores_C.csv')
    np.savetxt(output_path + r'\testScores_C_by_hour.csv', rmse_by_hour_test, delimiter=",")

    '''MAE'''
    mae_by_station_train.to_csv(output_path + r'\trainScores_MAE_C.csv')
    np.savetxt(output_path + r'\trainScores_MAE_C_by_hour.csv', mae_by_hour_train, delimiter=",")

    print('max test MAE', round(mae_by_station_test.max(), 2))
    print('min test MAE', round(mae_by_station_test.min(), 2))
    mae_by_station_test.to_csv(output_path + r'\testScores_MAE_C.csv')
    np.savetxt(output_path + r'\testScores_MAE_C_by_hour.csv', mae_by_hour_test, delimiter=",")