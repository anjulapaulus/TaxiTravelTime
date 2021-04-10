import math
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from Preprocessing import fill_zeros_bin
from Preprocessing import get_unique_bins
from Preprocessing import smooth


# Moving Averages
def moving_averages_r_prediction(ratio):
    pred_ratio = ratio['Ratios'].values[0]
    errors = []
    pred_val = []
    window_size = 3
    pred_ratio_val = []
    for i in range(0, 4464 * 30):
        if i % 4464 == 0:
            pred_ratio_val.append(0)
            pred_val.append(0)
            errors.append(0)
            continue
        pred_ratio_val.append(pred_ratio)
        pred_val.append(int((ratio['Given'].values[i]) * pred_ratio))
        errors.append(
            abs((math.pow(int((ratio['Given'].values[i]) * pred_ratio) - ratio['Prediction'].values[i], 1))))
        if i + 1 >= window_size:
            pred_ratio = sum(ratio['Ratios'].values[(i + 1) - window_size:(i + 1)]) / window_size
        else:
            pred_ratio = sum(ratio['Ratios'].values[0:(i + 1)]) / (i + 1)

    ratio['MAR_Predicted'] = pred_val
    ratio['MAR_Error'] = errors
    mape_error = (sum(errors) / len(errors)) / (sum(ratio['Prediction'].values) / len(ratio['Prediction'].values))
    mse_error = sum([e ** 2 for e in errors]) / len(errors)
    return ratio, mape_error, mse_error


def moving_average_p_prediction(ratio):
    predict_value = ratio['Prediction'].values[0]
    error = []
    pred_values = []
    window_size = 1
    for i in range(0, 4464 * 30):
        pred_values.append(predict_value)
        error.append(abs((math.pow(predict_value - ratio['Prediction'].values[i], 1))))
        if i + 1 >= window_size:
            predict_value = int(sum(ratio['Prediction'].values[(i + 1) - window_size:(i + 1)]) / window_size)
        else:
            predict_value = int(sum(ratio['Prediction'].values[0:(i + 1)]) / (i + 1))

    ratio['MAP_Predicted'] = pred_values
    ratio['MAP_Error'] = error
    mape_error = (sum(error) / len(error)) / (sum(ratio['Prediction'].values) / len(ratio['Prediction'].values))
    mse_error = sum([e ** 2 for e in error]) / len(error)
    return ratio, mape_error, mse_error


# Weighted Moving Averages

def weighted_ma_r_prediction(ratio):
    predicted_ratio = ratio['Ratios'].values[0]
    error = []
    predicted_values = []
    window_size = 5
    predicted_ratio_values = []
    for i in range(0, 4464 * 30):
        if i % 4464 == 0:
            predicted_ratio_values.append(0)
            predicted_values.append(0)
            error.append(0)
            continue
        predicted_ratio_values.append(predicted_ratio)
        predicted_values.append(int((ratio['Given'].values[i]) * predicted_ratio))
        error.append(
            abs((math.pow(int((ratio['Given'].values[i]) * predicted_ratio) - ratio['Prediction'].values[i], 1))))
        if i + 1 >= window_size:
            sum_values = 0
            sum_of_coeff = 0
            for j in range(window_size, 0, -1):
                sum_values += j * ratio['Ratios'].values[i - window_size + j]
                sum_of_coeff += j
            predicted_ratio = sum_values / sum_of_coeff
        else:
            sum_values = 0
            sum_of_coeff = 0
            for j in range(i + 1, 0, -1):
                sum_values += j * ratio['Ratios'].values[j - 1]
                sum_of_coeff += j
            predicted_ratio = sum_values / sum_of_coeff

    ratio['WAR_Predicted'] = predicted_values
    ratio['WAR_Error'] = error
    mape_err = (sum(error) / len(error)) / (sum(ratio['Prediction'].values) / len(ratio['Prediction'].values))
    mse_err = sum([e ** 2 for e in error]) / len(error)
    return ratio, mape_err, mse_err


def weighted_ma_p_prediction(ratio):
    predicted_value = ratio['Prediction'].values[0]
    error = []
    predicted_values = []
    window_size = 2
    for i in range(0, 4464 * 30):
        predicted_values.append(predicted_value)
        error.append(abs((math.pow(predicted_value - ratio['Prediction'].values[i], 1))))
        if i + 1 >= window_size:
            sum_values = 0
            sum_of_coeff = 0
            for j in range(window_size, 0, -1):
                sum_values += j * ratio['Prediction'].values[i - window_size + j]
                sum_of_coeff += j
            predicted_value = int(sum_values / sum_of_coeff)

        else:
            sum_values = 0
            sum_of_coeff = 0
            for j in range(i + 1, 0, -1):
                sum_values += j * ratio['Prediction'].values[j - 1]
                sum_of_coeff += j
            predicted_value = int(sum_values / sum_of_coeff)

    ratio['WA_P_Predicted'] = predicted_values
    ratio['WA_P_Error'] = error
    mape_err = (sum(error) / len(error)) / (sum(ratio['Prediction'].values) / len(ratio['Prediction'].values))
    mse_err = sum([e ** 2 for e in error]) / len(error)
    return ratio, mape_err, mse_err


# Exponential Weighted Averages

def exponential_war_predictions(ratios):
    predicted_ratio = ratios['Ratios'].values[0]
    alpha = 0.6
    error = []
    predicted_values = []
    predicted_ratio_values = []
    for i in range(0, 4464 * 30):
        if i % 4464 == 0:
            predicted_ratio_values.append(0)
            predicted_values.append(0)
            error.append(0)
            continue
        predicted_ratio_values.append(predicted_ratio)
        predicted_values.append(int((ratios['Given'].values[i]) * predicted_ratio))
        error.append(
            abs((math.pow(int((ratios['Given'].values[i]) * predicted_ratio) - ratios['Prediction'].values[i], 1))))
        predicted_ratio = (alpha * predicted_ratio) + (1 - alpha) * (ratios['Ratios'].values[i])

    ratios['EAR_Predicted'] = predicted_values
    ratios['EAR_Error'] = error
    mape_err = (sum(error) / len(error)) / (sum(ratios['Prediction'].values) / len(ratios['Prediction'].values))
    mse_err = sum([e ** 2 for e in error]) / len(error)
    return ratios, mape_err, mse_err


def exponential_wap_predictions(ratio):
    predicted_value = ratio['Prediction'].values[0]
    alpha = 0.3
    error = []
    predicted_values = []
    for i in range(0, 4464 * 30):
        if i % 4464 == 0:
            predicted_values.append(0)
            error.append(0)
            continue
        predicted_values.append(predicted_value)
        error.append(abs((math.pow(predicted_value - ratio['Prediction'].values[i], 1))))
        predicted_value = int((alpha * predicted_value) + (1 - alpha) * (ratio['Prediction'].values[i]))

    ratio['EAP_Predicted'] = predicted_values
    ratio['EAP_Error'] = error
    mape_err = (sum(error) / len(error)) / (sum(ratio['Prediction'].values) / len(ratio['Prediction'].values))
    mse_err = sum([e ** 2 for e in error]) / len(error)
    return ratio, mape_err, mse_err


def smoothing_bins():
    df_2013_08 = pd.read_csv("Datasets/preprocess_2013_08.csv")
    df_2014_08 = pd.read_csv("Datasets/preprocess_2014_08.csv")
    df_2014_09 = pd.read_csv("Datasets/preprocess_2014_09.csv")
    df_2014_10 = pd.read_csv("Datasets/preprocess_2014_10.csv")

    df_unique_2013_08 = get_unique_bins(df_2013_08)
    df_unique_2014_08 = get_unique_bins(df_2014_08)
    df_unique_2014_09 = get_unique_bins(df_2014_09)
    df_unique_2014_10 = get_unique_bins(df_2014_10)

    group_2013_08 = df_2013_08[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
        ['pickup_cluster', 'pickup_bins']).count()
    group_2014_08 = df_2014_08[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
        ['pickup_cluster', 'pickup_bins']).count()
    group_2014_09 = df_2014_09[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
        ['pickup_cluster', 'pickup_bins']).count()
    group_2014_10 = df_2014_10[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
        ['pickup_cluster', 'pickup_bins']).count()

    smooth_2013_08 = smooth(group_2013_08['trip_distance'].values, df_unique_2013_08)
    smooth_2014_08 = fill_zeros_bin(group_2014_08['trip_distance'].values, df_unique_2014_08)
    smooth_2014_09 = fill_zeros_bin(group_2014_09['trip_distance'].values, df_unique_2014_09)
    smooth_2014_10 = fill_zeros_bin(group_2014_10['trip_distance'].values, df_unique_2014_10)

    return smooth_2013_08, smooth_2014_08, smooth_2014_09, smooth_2014_10


if __name__ == '__main__':
    smooth_2013_08, smooth_2014_08, smooth_2014_09, smooth_2014_10 = smoothing_bins()
    ratios = pd.DataFrame()
    ratios['Given'] = smooth_2013_08
    ratios['Prediction'] = smooth_2014_08
    ratios['Ratios'] = ratios['Prediction'] * 1.0 / ratios['Given'] * 1.0

    mean_err = [0] * 6
    median_err = [0] * 6
    ratios, mean_err[0], median_err[0] = moving_averages_r_prediction(ratios)
    ratios1, mean_err[1], median_err[1] = moving_average_p_prediction(ratios)
    ratios2, mean_err[2], median_err[2] = weighted_ma_r_prediction(ratios)
    ratios3, mean_err[3], median_err[3] = weighted_ma_p_prediction(ratios)
    ratios4, mean_err[4], median_err[4] = exponential_war_predictions(ratios)
    ratios5, mean_err[5], median_err[5] = exponential_wap_predictions(ratios)
    print("Error Metric Matrix (Forecasting Methods) - MAPE & MSE")
    print("--------------------------------------------------------------------------------------------------------")
    print("Moving Averages (Ratios) -                       MAPE: ", mean_err[0], "      MSE: ", median_err[0])
    print("Moving Averages (2014 Values) -                  MAPE: ", mean_err[1], "      MSE: ", median_err[1])
    print("--------------------------------------------------------------------------------------------------------")
    print("Weighted Moving Averages (Ratios) -              MAPE: ", mean_err[2], "      MSE: ", median_err[2])
    print("Weighted Moving Averages (2014 Values) -         MAPE: ", mean_err[3], "      MSE: ", median_err[3])
    print("--------------------------------------------------------------------------------------------------------")
    print("Exponential Moving Averages (Ratios) -           MAPE: ", mean_err[4], "      MSE: ", median_err[4])
    print("Exponential Moving Averages (2014 Values) -      MAPE: ", mean_err[5], "      MSE: ", median_err[5])
    print("--------------------------------------------------------------------------------------------------------")
