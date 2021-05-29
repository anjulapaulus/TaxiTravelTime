import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import mean
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split, RepeatedKFold, cross_val_score, \
    RepeatedStratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import catboost as cb
from time import time

desired_width = 320

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


def linear_regression(train, test, train_output, test_output):
    # model = LinearRegression()
    # # define evaluation
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # space = dict()
    # space['fit_intercept'] = [True, False]
    # space['normalize'] = [True, False]
    #
    # search = GridSearchCV(model, space, scoring='neg_mean_absolute_error', n_jobs=-1, cv=cv)
    # result = search.fit(train, train_output)
    # # summarize result
    # print('Best Score: %s' % result.best_score_)
    # print('Best Hyperparameters: %s' % result.best_params_)

    # applying linear regress ion with best hyper-parameter
    lin_reg = LinearRegression(normalize=False, fit_intercept=True)
    t0 = time()
    lin_reg.fit(train, train_output)
    print('Linear Training Time:', (time() - t0))

    pkl_file = "Models/linear_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(lin_reg, file)


def random_forest_regression(train, test, train_output, test_output):
    # Hyper parameter tuning
    # params = {
    #     'n_estimators': [600, 800, 1000, 1500, 2500],
    #     'max_features': ['auto', 'sqrt'],
    #     'min_samples_leaf': [1, 2, 5, 10, 15],
    #     'min_samples_split': [2, 5, 10, 15, 20],
    #     'max_depth': [10, 20, 30, 40, 50],
    # }
    # rf = RandomForestRegressor()
    # #
    # grid_search = GridSearchCV(estimator=rf, param_grid=params,
    #                            cv=3, n_jobs=-1, verbose=2)
    # grid_search.fit(train, train_output)
    # print(grid_search.best_params_)

    rfr = RandomForestRegressor(n_estimators=1000)
    t0 = time()
    rfr.fit(train, train_output)
    print('Random FOrest Training Time:', (time() - t0))
    pkl_file = "Models/random_forest_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(rfr, file)


def xgb_regression(x_train, x_test, y_train, y_test):
    # Hyper parameter tuning
    # params = {
    #     'n_estimators': [300, 350, 400, 450, 500],
    #     'max_depth': [2, 3, 4, 5]
    # }
    #
    # model = xgb.XGBRegressor()
    # grid = GridSearchCV(model, params, cv=2, n_jobs=5, verbose=True)
    # grid.fit(x_train, y_train)
    # print(grid.best_score_)
    # print(grid.best_params_)

    xgbReg = xgb.XGBRegressor(max_depth=5, n_estimators=350, verbosity=3, eval_metric="mae")
    t0 = time()
    xgbReg.fit(x_train, y_train)
    # y_pred = xgbReg.predict(x_test)
    print('XGB Training Time:', (time() - t0))
    pkl_file = "Models/xgb_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(xgbReg, file)


def catboost_regression(x_train, x_test, y_train, y_test):
    params = {'iterations': [500, 600, 650],
              'learning_rate': [0.1, 0.2, 0.3],
              'depth': [2, 4, 6, 8],
              'l2_leaf_reg': [0.2, 0.5, 1]}
    model = cb.CatBoostRegressor()

    random = model.grid_search(params, X=x_train, y=y_train, verbose=True)
    print(random)

    # cat = cb.CatBoostRegressor(depth=8, iterations=650,
    #                            learning_rate=0.3,
    #                            l2_leaf_reg=0.5)
    # t0 = time()
    # cat.fit(X=x_train, y=y_train)
    # print('Catboost Training Time:', (time() - t0))
    # pkl_file = "Models/catboost_regression.pkl"
    #
    # with open(pkl_file, 'wb') as file:
    #     pickle.dump(cat, file)

# def light_gbm_regression(x_train, x_test, y_train, y_test):
#     lightgbm.Re
def mape(actual, predicted):
    m = np.mean(np.abs((actual - predicted) / actual)) * 100
    return m


def get_stacking(x_train, x_test, y_train, y_test):

    # define the base models
    level0 = list()
    level0.append(('xgb', xgb.XGBRegressor(max_depth=5, n_estimators=350, verbosity=3, eval_metric="mae")))
    level0.append(('cat', cb.CatBoostRegressor(depth=8, iterations=650,
                                               learning_rate=0.3,
                                               l2_leaf_reg=0.5)))

    # define meta learner model
    level1 = LinearRegression(normalize=False, fit_intercept=True)
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)

    model.fit(x_train, y_train)
    pkl_file = "Models/stacker1.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(model, file)



def kfold_run(X, y):
    mseAr = []
    rmseAr = []
    maeAr = []
    mapeAr = []
    spearmanAr = []

    num_x = X.to_numpy()
    num_y = y.to_numpy()
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)
    for X_train_index, X_test_index in kfold.split(num_x):
        X_train, X_test = num_x[X_train_index], num_x[X_test_index]
        y_train, y_test = num_y[X_train_index], num_y[X_test_index]

        file = open("Models/stacker.pkl", 'rb')
        model = pickle.load(file)
        # file = open("Models/xgb_regression.pkl", 'rb')
        # model = pickle.load(file)
        predictedCAT = model.predict(X_test)
        mseCAT = mean_squared_error(y_test, predictedCAT)
        rmseCAT = mean_squared_error(y_test, predictedCAT, squared=False)
        maeCAT = mean_absolute_error(y_test, predictedCAT)
        mapeCAT = mape(y_test, predictedCAT)
        spearmanCAT, p = spearmanr(y_test, predictedCAT)
        mseAr.append(mseCAT)
        rmseAr.append(rmseCAT)
        maeAr.append(maeCAT)
        mapeAr.append(mapeCAT)
        spearmanAr.append(spearmanCAT)

    # print('Catboost Metrics')
    # print('........................')
    # print('MSE: ', mean(mseAr))
    # print('RMSE: ', mean(rmseAr))
    # print('MAE: ', mean(maeAr))
    # print('MAPE: ', mean(mapeAr))
    # print('Spearman Correlation: ', mean(spearmanAr))
    # print('XGBoost Metrics')
    # print('........................')
    # print('MSE: ', mean(mseAr))
    # print('RMSE: ', mean(rmseAr))
    # print('MAE: ', mean(maeAr))
    # print('MAPE: ', mean(mapeAr))
    # print('Spearman Correlation: ', mean(spearmanAr))
    print('Stacker Metrics')
    print('........................')
    print('MSE: ', mean(mseAr))
    print('RMSE: ', mean(rmseAr))
    print('MAE: ', mean(maeAr))
    print('MAPE: ', mean(mapeAr))
    print('Spearman Correlation: ', mean(spearmanAr))


def kfold_run_train_catboost(X, y):
    mseAr = []
    rmseAr = []
    maeAr = []
    mapeAr = []
    spearmanAr = []

    num_x = X.to_numpy()
    num_y = y.to_numpy()
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)
    cat = cb.CatBoostRegressor(depth=8, iterations=650,
                               learning_rate=0.3,
                               l2_leaf_reg=0.5)

    for X_train_index, X_test_index in kfold.split(num_x):
        X_train, X_test = num_x[X_train_index], num_x[X_test_index]
        y_train, y_test = num_y[X_train_index], num_y[X_test_index]
        cat.fit(X=X_train, y=y_train)

    pkl_file = "Models/catboost_regression1.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(cat, file)

def kfold_run_train_xgboost(X, y):

    num_x = X.to_numpy()
    num_y = y.to_numpy()
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)
    xgbReg = xgb.XGBRegressor(max_depth=5, n_estimators=350, verbosity=3, eval_metric="mae")

    for X_train_index, X_test_index in kfold.split(num_x):
        X_train, X_test = num_x[X_train_index], num_x[X_test_index]
        y_train, y_test = num_y[X_train_index], num_y[X_test_index]
        xgbReg.fit(X_train, y_train)

    pkl_file = "Models/xgb_regression1.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(xgbReg, file)

def kfold_run_train_stacker(X, y):

    num_x = X.to_numpy()
    num_y = y.to_numpy()
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)

    level0 = list()
    level0.append(('xgb', xgb.XGBRegressor(max_depth=5, n_estimators=350, verbosity=3, eval_metric="mae")))
    level0.append(('cat', cb.CatBoostRegressor(depth=8, iterations=650,
                                               learning_rate=0.3,
                                               l2_leaf_reg=0.5)))

    # define meta learner model
    level1 = LinearRegression()
    # define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)

    for X_train_index, X_test_index in kfold.split(num_x):
        X_train, X_test = num_x[X_train_index], num_x[X_test_index]
        y_train, y_test = num_y[X_train_index], num_y[X_test_index]
        model.fit(X_train, y_train)

    pkl_file = "Models/stacker1.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    df = pd.read_csv('Datasets/preprocessed.csv')
    # print(df.columns)
    x_data = df[
        ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'hour_of_day', 'week_day',
         'pickup_cluster', 'dropoff_cluster', 'osrm_distance', 'osrm_duration', 'steps', 'intersections', 'bin']]
    y_data = df['trip_duration']

    # kfold_run(x_data, y_data)
    # kfold_run_train_xgboost(x_data, y_data)
    # kfold_run_train_stacker(x_data, y_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, shuffle=True)

    # xgb_regression(x_train, x_test, y_train, y_test)
    # catboost_regression(x_train, x_test, y_train, y_test)
    # random_forest_regression(x_train, x_test, y_train, y_test)
    # linear_regression(x_train, x_test, y_train, y_test)
    # svr_linear(x_train, x_test, y_train, y_test)
    # sgd_regressor(x_train, x_test, y_train, y_test)
    # get_stacking(x_train, x_test, y_train, y_test)
    #
    file = open("Models/catboost_regression.pkl", 'rb')
    model = pickle.load(file)
    predictedCAT = model.predict(x_test)
    mseCAT = mean_squared_error(y_test, predictedCAT)
    rmseCAT = mean_squared_error(y_test, predictedCAT, squared=False)
    maeCAT = mean_absolute_error(y_test, predictedCAT)
    mapeCAT = mape(y_test, predictedCAT)
    spearmanCAT = spearmanr(y_test, predictedCAT)
    pearson = pearsonr(y_test, predictedCAT)
    print('CatBoost Metrics')
    print('........................')
    print('MSE: ', mseCAT)
    print('RMSE: ', rmseCAT)
    print('MAE: ', maeCAT)
    print('MAPE: ', mapeCAT)
    print('Spearman Correlation: ', spearmanCAT)
    print('Pearson Correlation: ', pearson)
    print()

    # fileXGB = open("Models/xgb_regression.pkl", 'rb')
    # modelXGB = pickle.load(fileXGB)
    # predictedXGB = modelXGB.predict(x_test)
    # mseXGB = mean_squared_error(y_test, predictedXGB)
    # rmseXGB = mean_squared_error(y_test, predictedXGB, squared=False)
    # maeXGB = mean_absolute_error(y_test, predictedXGB)
    # mapeXGB = mape(y_test, predictedXGB)
    # spearmanXGB = spearmanr(y_test, predictedXGB)
    # print('XGBoost Metrics')
    # print('........................')
    # print('MSE: ', mseXGB)
    # print('RMSE: ', rmseXGB)
    # print('MAE: ', maeXGB)
    # print('MAPE: ', mapeXGB)
    # print('Spearman Correlation: ', spearmanXGB)
    # print()
    # #
    # fileLinear = open("Models/linear_regression.pkl", 'rb')
    # modelLinear = pickle.load(fileLinear)
    # predictedL = modelLinear.predict(x_test)
    # mseL = mean_squared_error(y_test, predictedL)
    # rmseL = mean_squared_error(y_test, predictedL, squared=False)
    # maeL = mean_absolute_error(y_test, predictedL)
    # mapeL = mape(y_test, predictedL)
    # spearmanL = spearmanr(y_test, predictedL)
    # print('Linear Metrics')
    # print('........................')
    # print('MSE: ', mseL)
    # print('RMSE: ', rmseL)
    # print('MAE: ', maeL)
    # print('MAPE: ', mapeL)
    # print('Spearman Correlation: ', spearmanL)
    # print()

    # fileLinear = open("Models/stacker.pkl", 'rb')
    # modelStacking = pickle.load(fileLinear)
    # predictedS = modelStacking.predict(x_test)
    # mseS = mean_squared_error(y_test, predictedS)
    # rmseS = mean_squared_error(y_test, predictedS, squared=False)
    # maeS = mean_absolute_error(y_test, predictedS)
    # mapeS = mape(y_test, predictedS)
    # spearmanS = spearmanr(y_test, predictedS)
    # print('Stacking Metrics')
    # print('........................')
    # print('MSE: ', mseS)
    # print('RMSE: ', rmseS)
    # print('MAE: ', maeS)
    # print('MAPE: ', mapeS)
    # print('Spearman Correlation: ', spearmanS)
    # print()


