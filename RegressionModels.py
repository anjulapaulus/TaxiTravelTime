import pandas as pd
import numpy as np
import pickle
import math

from numpy import absolute
from scipy.stats import spearmanr
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import cov
import xgboost as xgb
import catboost as cb

desired_width = 320


# def prepare(df):
#     df = pd.concat([df, pd.get_dummies(df['hour_of_day'], prefix='hrs')], axis=1)
#     # df = pd.concat([df, pd.get_dummies(df['week_day'], prefix='day')], axis=1)
#     # df = pd.concat([df, pd.get_dummies(df['clusters'], prefix='route')], axis=1)
#     # df.drop(df.columns[[0, 1, 2, 3, 4, 5, 8, 16, 17, 18, 19, 20]], axis=1, inplace=True)
#     print(df.head())


def linear_regression(train, test, train_output, test_output):
    std_train = StandardScaler().fit_transform(train)
    std_test = StandardScaler().fit_transform(test)
    # parameter tuning
    classifier_Lr = SGDRegressor(loss='squared_loss', penalty='l2')
    lambda_val = [10 ** -14, 10 ** -12, 10 ** -10, 10 ** -8, 10 ** -6,
                  10 ** -4, 10 ** -2, 10 ** 0, 10 ** 2, 10 ** 4, 10 ** 6]
    HP = {'alpha': lambda_val}
    # 3fold cross-validation
    grid_param = GridSearchCV(classifier_Lr, HP,
                              scoring='neg_mean_absolute_error', cv=3)
    grid_param.fit(std_train, train_output)
    best_alpha = grid_param.best_params_['alpha']

    # applying linear regress ion with best hyper-parameter
    sgd = SGDRegressor(loss="squared_loss", penalty="l2", alpha=best_alpha)
    sgd.fit(std_train, train_output)

    pkl_file = "Models/linear_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(sgd, file)


def random_forest_regression(train, test, train_output, test_output):
    # Hyper parameter tuning
    # params = {
    #     'bootstrap': [True, False],
    #     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     'max_features': ['auto', 'sqrt'],
    #     'min_samples_leaf': [1, 2, 4],
    #     'min_samples_split': [2, 5, 10],
    #     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #
    # }
    # rf = RandomForestRegressor()
    #
    # grid_search = GridSearchCV(estimator=rf, param_grid=params,
    #                            cv=3, n_jobs=-1, verbose=2)
    # grid_search.fit(train, train_output)

    rfr = RandomForestRegressor(max_features='sqrt', min_samples_leaf=4, min_samples_split=3, n_estimators=800,
                                n_jobs=-1)
    rfr.fit(train, train_output)

    pkl_file = "Models/random_forest_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(rfr, file)


def xgb_regression(x_train, x_test, y_train, y_test):
    # Hyper parameter tuning
    # params = {
    #     'n_estimators': [300, 350, 400, 450, 500],
    #     'max_depth': [2, 3, 4, 5]
    # }

    # model = xgb.XGBRegressor()
    # grid = GridSearchCV(model, params, cv=2, n_jobs=5, verbose=True)
    # grid.fit(x_train, y_train)
    # print(grid.best_score_)
    # print(grid.best_params_)

    xgbReg = xgb.XGBRegressor(max_depth=5, n_estimators=350, verbosity=3, eval_metric="mae")

    xgbReg.fit(x_train, y_train)
    # y_pred = xgbReg.predict(x_test)

    pkl_file = "Models/xgb_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(xgbReg, file)


def catboost_regression(x_train, x_test, y_train, y_test):
    # params = {'iterations': [500, 600, 650],
    #           'learning_rate': [0.1, 0.2, 0.3],
    #           'depth': [2, 4, 6, 8],
    #           'l2_leaf_reg': [0.2, 0.5, 1]}
    # model = cb.CatBoostRegressor()
    #
    # random = model.grid_search(params, X=x_train, y=y_train, verbose=True)
    # print(random)

    cat = cb.CatBoostRegressor(depth=8, iterations=650,
                               learning_rate=0.3,
                               l2_leaf_reg=0.5)

    cat.fit(X=x_train, y=y_train)
    pkl_file = "Models/catboost_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(cat, file)


# def kFold(model, x, y):
#     model.predict()


# def meta_learning_test():


if __name__ == '__main__':
    # pd.set_option('display.width', desired_width)
    #
    # np.set_printoptions(linewidth=desired_width)
    #
    # pd.set_option('display.max_columns', 20)
    # split()
    df = pd.read_csv('Datasets/preprocessed.csv')
    # print(df.columns)
    x_data = df[
        ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'hour_of_day', 'week_day',
         'pickup_cluster', 'dropoff_cluster', 'osrm_distance', 'osrm_duration', 'steps', 'intersections', 'bin']]
    y_data = df['trip_duration']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)

    # xgb_regression(x_train, x_test, y_train, y_test)
    # catboost_regression(x_train, x_test, y_train, y_test)
    random_forest_regression(x_train, x_test, y_train, y_test)
    # linear_regression(x_train, x_test, y_train, y_test)

    # file = open("Models/catboost_regression.pkl", 'rb')
    # model = pickle.load(file)
    # predictedCAT = model.predict(x_test)
    # mseCAT = mean_squared_error(y_test, predictedCAT)
    # rmseCAT = mean_squared_error(y_test, predictedCAT, squared=False)
    # maeCAT = mean_absolute_error(y_test, predictedCAT)
    # print('CatBoost Metrics')
    # print('........................')
    # print('MSE: ', mseCAT)
    # print('RMSE: ', rmseCAT)
    # print('MAE: ', maeCAT)
    # print()
    # fileXGB = open("Models/xgb_regression.pkl", 'rb')
    # modelXGB = pickle.load(fileXGB)
    # predictedXGB = modelXGB.predict(x_test)
    # mseXGB = mean_squared_error(y_test, predictedXGB)
    # rmseXGB = mean_squared_error(y_test, predictedXGB, squared=False)
    # maeXGB = mean_absolute_error(y_test, predictedXGB)
    # print('XGBoost Metrics')
    # print('........................')
    # print('MSE: ', mseXGB)
    # print('RMSE: ', rmseXGB)
    # print('MAE: ', maeXGB)
    # print()
    # fileLinear = open("Models/xgb_regression.pkl", 'rb')
    # modelLinear = pickle.load(fileLinear)
    # predictedL = modelLinear.predict(x_test)
    # mseL = mean_squared_error(y_test, predictedL)
    # rmseL = mean_squared_error(y_test, predictedL, squared=False)
    # maeL = mean_absolute_error(y_test, predictedL)
    # print('Linear Metrics')
    # print('........................')
    # print('MSE: ', mseL)
    # print('RMSE: ', rmseL)
    # print('MAE: ', maeL)
