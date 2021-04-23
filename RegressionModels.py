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
from sklearn.metrics import mean_absolute_error
from numpy import cov
import xgboost as xgb
import catboost as cb

desired_width = 320


def prepare(df):
    df = pd.concat([df, pd.get_dummies(df['hour_of_day'], prefix='hrs')], axis=1)
    # df = pd.concat([df, pd.get_dummies(df['week_day'], prefix='day')], axis=1)
    # df = pd.concat([df, pd.get_dummies(df['clusters'], prefix='route')], axis=1)
    # df.drop(df.columns[[0, 1, 2, 3, 4, 5, 8, 16, 17, 18, 19, 20]], axis=1, inplace=True)
    print(df.head())


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
    clf = SGDRegressor(loss="squared_loss", penalty="l2", alpha=best_alpha)
    clf.fit(std_train, train_output)

    # y_pred = clf.predict(std_test)
    # lr_test_predictions = [round(value) for value in y_pred]
    # y_pred = clf.predict(std_train)
    # lr_train_predictions = [round(value) for value in y_pred]

    # print((mean_absolute_error(train_output, lr_train_predictions)) / (sum(train_output) / len(train_output)))
    # print((mean_absolute_error(test_output, lr_test_predictions)) / (sum(test_output) / len(test_output)))

    pkl_file = "linear_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(clf, file)


def random_forest_regression(train, test, train_output, test_output):
    # Hyper parameter tuning
    C = [10, 20, 40]
    random_clf = RandomForestRegressor(n_jobs=-1)
    HP1 = {'n_estimators': C}
    n_iter1 = 10
    # 3fold cross-validation
    grid_param1 = RandomizedSearchCV(random_clf, HP1,
                                     scoring='neg_mean_absolute_error', cv=3,
                                     n_iter=n_iter1)
    grid_param1.fit(train, train_output)
    best_alpha1 = grid_param1.best_params_['n_estimators']

    clf1 = RandomForestRegressor(max_features='sqrt', min_samples_leaf=4, min_samples_split=3, n_estimators=best_alpha1,
                                 n_jobs=-1)
    clf1.fit(train, train_output)

    # y_pred = clf1.predict(test)
    # rndf_test_predictions = [round(value) for value in y_pred]
    # y_pred = clf1.predict(train)
    # rndf_train_predictions = [round(value) for value in y_pred]
    #
    # print('Random Forest Train MAE: ', (mean_absolute_error(train_output, rndf_train_predictions)) / (sum(train_output) / len(train_output)))
    # print('Random Forest Test MAE: ', (mean_absolute_error(test_output, rndf_test_predictions)) / (sum(test_output) / len(test_output)))

    pkl_file = "random_forest_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(clf1, file)


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

    pkl_file = "xgb_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(xgbReg, file)

    xg_MAE = []
    xg_RMSE = []
    xg_MAPE = []
    y_pred = []

    folds = 5
    for k in range(folds):
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        xg = xgbReg
        xg.fit(x_train, y_train)
        y_pred = xg.predict(x_test)
        xg_MAE.append(mae(y_test, y_pred))
        xg_RMSE.append(rmse(y_test, y_pred))
        xg_MAPE.append(mape(y_test, y_pred))

    print('Metrics for XGB')
    print('_______________________________________')
    print('MAE : ', np.mean(xg_MAE))
    print('RMSE : ', np.mean(xg_RMSE))
    print('MAPE : ', np.mean(xg_MAPE))
    print('Spearman Cofficient: ', spearmanr(y_pred, y_test))

    pkl_file = "xgb_regression.pkl"

    with open(pkl_file, 'wb') as file:
        pickle.dump(xgbReg, file)


# Calculate mean absolute error
def mae(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += abs(predicted[i] - actual[i])
    return sum_error / float(len(actual))


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100


def rmse(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return math.sqrt(mean_error)


def catboost_regression(x_train, x_test, y_train, y_test):
    params = {'iterations': [400, 450, 500, 600, 650],
              'learning_rate': [0.1, 0.2, 0.3],
              'depth': [2, 4, 6, 8],
              'l2_leaf_reg': [0.2, 0.5, 1, 3]}
    model = cb.CatBoostRegressor()

    # random = model.grid_search(grid, X=train, y=train_output)

    grid = GridSearchCV(model, params, cv=2, n_jobs=5, verbose=True)
    grid.fit(x_train, y_train)
    print(grid.best_score_)
    print(grid.best_params_)

    # clf = cb.CatBoostRegressor(depth=random['params']['depth'], iterations=random['params']['iterations'],
    #                            learning_rate=random['params']['learning_rate'],
    #                            l2_leaf_reg=random['params']['l2_leaf_reg'])
    #
    # clf.fit(X=train, y=train_output)
    # pkl_file = "catboost_regression.pkl"
    #
    # with open(pkl_file, 'wb') as file:
    #     pickle.dump(clf, file)
    # y_pred = clf.predict(test)
    # # {'depth': 6, 'iterations': 200, 'learning_rate': 0.1, 'l2_leaf_reg': 0.5}
    # cat_test_predictions = [round(value) for value in y_pred]
    # y_pred = clf.predict(train)
    # cat_train_predictions = [round(value) for value in y_pred]
    # print('CatBoost Train MAE: ', (mean_absolute_error(train_output, cat_train_predictions)) / (sum(train_output) / len(train_output)))
    # print('CatBoost Test MAE: ', (mean_absolute_error(test_output, cat_test_predictions)) / (sum(test_output) / len(test_output)))
    # trainMAE = (mean_absolute_error(train_output, cat_train_predictions)) / (sum(train_output) / len(train_output))
    # testMAE = (mean_absolute_error(test_output, cat_test_predictions)) / (sum(test_output) / len(test_output))
    # trainMSE = mean_squared_error(train_output, cat_train_predictions)
    # trainRMSE = math.sqrt(trainMSE)
    # testMSE = mean_squared_error(test_output, cat_test_predictions)
    # testRMSE = math.sqrt(testMSE)
    # trainMAPE = mean_absolute_percentage_error(train_output, cat_train_predictions)
    # testMAPE = mean_absolute_percentage_error(test_output, cat_test_predictions)
    # trainSpearman = spearmanr(train_output, cat_train_predictions)
    # testSpearman = spearmanr(train_output, cat_train_predictions)
    # print('Train MAE: ', trainMAE)
    # print('Test MAE: ', testMAE)
    # print('Train RMSE: ', trainRMSE)
    # print('Test RMSE: ', testRMSE)

    # random = model.predict(X_test)
    # rmse = (np.sqrt(mean_squared_error(y_test, pred)))
    # r2 = r2_score(y_test, pred)
    # print("Testing performance")
    # print('RMSE: {:.2f}'.format(rmse))
    # print('R2: {:.2f}'.format(r2))
    # 0.20153229645731532
    # 0.17286462748497383


# def meta_learning():


if __name__ == '__main__':
    pd.set_option('display.width', desired_width)

    np.set_printoptions(linewidth=desired_width)

    pd.set_option('display.max_columns', 20)
    # split()
    df = pd.read_csv('Datasets/preprocessed.csv')
    # print(df.columns)
    x_data = df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'week_day', 'bin',
                 'pickup_cluster', 'dropoff_cluster', 'osrm_distance', 'osrm_duration', 'steps', 'intersections']]
    y_data = df['trip_duration']

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)

    xgb_regression(x_train, x_test, y_train, y_test)
    # catboost_regression(x_train, x_test, y_train, y_test)
