import pandas as pd
import numpy as np
from BaseLineModels import get_unique_bins
from BaseLineModels import smooth
from BaseLineModels import fill_zeros_bin
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RandomizedSearchCV

time_stamps = 5

outputList = []

lat = []
lon = []
weekday = []
tsne_feature = []
tsne_feature = [0] * time_stamps

alpha = 0.3
predicted_values = []
predict_list = []
tsne_flat_exp_avg = []


def prepare_for_split(regions_cumulative, kmeans):
    global tsne_feature, predicted_values, predicted_value
    final_df = pd.DataFrame(columns=['f_1', 'a_1', 'f_2', 'a_2', 'f_3', 'a_3', 'f_4', 'a_4', 'f_5', 'a_5'])

    for i in range(0, 30):
        # amplitutude €& frequecy are calcualted &  saved in dataframe
        aug_amp = np.fft.fft(np.array(regions_cumulative[i][0:4464]))
        aug_freq = np.fft.fftfreq(4464, 1)

        sep_amp = list(np.fft.fft(np.array(regions_cumulative[i])[4464:(4464 + 4464)]))
        sep_freq = list(np.fft.fftfreq(4464, 1))

        oct_amp = list(np.fft.fft(np.array(regions_cumulative[i])[(4464 + 4464):(4320 + 4464 + 4464)]))
        oct_freq = list(np.fft.fftfreq(4320, 1))

        aug_df = pd.DataFrame(data=aug_freq, columns=['Freq'])
        aug_df = pd.DataFrame(data=aug_amp, columns=['Amp'])
        sep_df = pd.DataFrame(data=sep_freq, columns=['Freq'])
        sep_df = pd.DataFrame(data=sep_amp, columns=['Amp'])
        oct_df = pd.DataFrame(data=oct_freq, columns=['Freq'])
        oct_df = pd.DataFrame(data=oct_amp, columns=['Amp'])

        aug_df_list = []
        sep_df_list = []
        oct_df_list = []

        aug_df_sort = aug_df.sort_values(by=['Amp'], ascending=False)[:5].reset_index(drop=True).T
        sep_df_sort = sep_df.sort_values(by=['Amp'], ascending=False)[:5].reset_index(drop=True).T
        oct_df_sort = oct_df.sort_values(by=['Amp'], ascending=False)[:5].reset_index(drop=True).T
        # print(aug_df_sort['Amp'][0])
        # print(type(fram_jan_sort['Freq'][0]))
        for j in range(0, 5):
            aug_df_list.append(float(aug_df_sort[j]))
            aug_df_list.append(float(aug_df_sort[j]))

            sep_df_list.append(float(sep_df_sort[j]))
            sep_df_list.append(float(sep_df_sort[j]))

            oct_df_list.append(float(oct_df_sort[j]))
            oct_df_list.append(float(oct_df_sort[j]))

        data1 = [aug_df_list] * 4464
        data2 = [sep_df_list] * 4464
        data3 = [oct_df_list] * 4320

        column_names = ['f_1', 'a_1', 'f_2', 'a_2', 'f_3', 'a_3', 'f_4', 'a_4', 'f_5', 'a_5']
        aug_df_new = pd.DataFrame(data=data1, columns=column_names)
        sep_df_new = pd.DataFrame(data=data2, columns=column_names)
        oct_df_new = pd.DataFrame(data=data3, columns=column_names)

        final_df = final_df.append(aug_df_new, ignore_index=True)
        final_df = final_df.append(sep_df_new, ignore_index=True)
        final_df = final_df.append(oct_df_new, ignore_index=True)

        lat.append([kmeans.cluster_centers_[i][0]] * 13243)  # 4464+4464+4320-5
        lon.append([kmeans.cluster_centers_[i][1]] * 13243)

        # aug 1st 2014 is friday, so we start our day from 4: "(int(k/144))%7+5"
        # our prediction start from 5th 10min intravel since we need to have number of pickups that are happened in last 5 pickup bins
        weekday.append([int(((int(k / 144)) % 7 + 5) % 7) for k in range(5, 4464 + 4464 + 4320)])

        # regions_cum is a list of lists [[x1,x2,x3..x13104], [x1,x2,x3..x13104], [x1,x2,x3..x13104], [x1,x2,x3..x13104], [x1,x2,x3..x13104], .. 40 lsits]
        tsne_feature = np.vstack((tsne_feature, [regions_cumulative[i][r:r + time_stamps] for r in
                                                 range(0, len(regions_cumulative[i]) - time_stamps)]))
        outputList.append(regions_cumulative[i][5:])

    tsne_feature = tsne_feature[1:]

    final_df.drop(['f_1'], axis=1, inplace=True)

    final_df = final_df
    final_df = final_df.fillna(0)
    # print(final_df.head(1))
    # print(tsne_feature.shape)
    # print(final_df.shape)
    # print(len(lat[0]) * len(lon) == tsne_feature.shape[0] == len(weekday) * len(weekday[0]) == 30 * 13243 == len(
    #     outputList) * len(outputList[0]))

    for r in range(0, 30):
        for i in range(0, 13248):
            if i == 0:
                predicted_value = regions_cumulative[r][0]
                predicted_values.append(0)
                continue
            predicted_values.append(predicted_value)
            predicted_value = int((alpha * predicted_value) + (1 - alpha) * (regions_cumulative[r][i]))
        predict_list.append(predicted_values[5:])
        predicted_values = []

    # print("size of train data :", int(13243 * 0.75))
    # print("size of test data :", int(13243 * 0.25))
    # size of train data : 9932
    # size of test data : 3310

    train_features = [tsne_feature[i * 13243:(13243 * i + 9932)] for i in range(0, 30)]
    test_features = [tsne_feature[(13243 * i) + 9932:13243 * (i + 1)] for i in range(0, 30)]

    # print(train_features[0])
    # print(test_features[0])
    final_train_df = pd.DataFrame(columns=['a_1', 'f_2', 'a_2', 'f_3', 'a_3',
                                           'f_4', 'a_4', 'f_5', 'a_5'])
    final_test_df = pd.DataFrame(columns=['a_1', 'f_2', 'a_2', 'f_3', 'a_3',
                                          'f_4', 'a_4', 'f_5', 'a_5'])
    for i in range(0, 30):
        # print(fram_final[i*13099:(13099*i+9824)])
        final_train_df = final_train_df.append(final_df[i * 13243:(13243 * i + 9932)])
    final_train_df.reset_index(inplace=True)

    for i in range(0, 30):
        # print(fram_final[(13099*(i))+9824:13099*(i+1)])
        final_test_df = final_test_df.append(final_df[(13243 * i) + 9932:13243 * (i + 1)])
    final_test_df.reset_index(inplace=True)

    final_test_df.drop(['index'], axis=1, inplace=True)
    final_train_df.drop(['index'], axis=1, inplace=True)

    # print(final_train_df.head(1)) #fine
    # print(final_test_df.head(1))

    # print("Number of data clusters", len(train_features), "Number of data points in trian data", len(train_features[0]),
    #       "Each data point contains", len(train_features[0][0]), "features")
    # print("Number of data clusters", len(train_features), "Number of data points in test data", len(test_features[0]),
    #       "Each data point contains", len(test_features[0][0]), "features")

    # extracting first 9932 timestamp values i.e 75% of 13243 (total timestamps) for our training data
    train_flat_lat = [i[:9932] for i in lat]
    train_flat_lon = [i[:9932] for i in lon]
    train_flat_weekday = [i[:9932] for i in weekday]
    train_flat_output = [i[:9932] for i in outputList]
    train_flat_exp_avg = [i[:9932] for i in predict_list]

    # extracting the rest of the timestamp values i.e 25% of 13243 (total timestamps) for our test data
    test_flat_lat = [i[9932:] for i in lat]
    test_flat_lon = [i[9932:] for i in lon]
    test_flat_weekday = [i[9932:] for i in weekday]
    test_flat_output = [i[9932:] for i in outputList]
    test_flat_exp_avg = [i[9932:] for i in predict_list]

    # the above contains values in the form of list of lists
    # (i.e. list of values of each region), here we make all of them in one list
    train_new_features = []
    for i in range(0, 30):
        train_new_features.extend(train_features[i])
    test_new_features = []
    for i in range(0, 30):
        test_new_features.extend(test_features[i])

    # converting lists of lists into sinle list i.e flatten
    # a  = [[1,2,3,4],[4,6,7,8]]
    # print(sum(a,[]))
    # [1, 2, 3, 4, 4, 6, 7, 8]

    train_lat = sum(train_flat_lat, [])
    train_lon = sum(train_flat_lon, [])
    train_weekday = sum(train_flat_weekday, [])
    train_output = sum(train_flat_output, [])
    train_exp_avg = sum(train_flat_exp_avg, [])

    test_lat = sum(test_flat_lat, [])
    test_lon = sum(test_flat_lon, [])
    test_weekday = sum(test_flat_weekday, [])
    test_output = sum(test_flat_output, [])
    test_exp_avg = sum(test_flat_exp_avg, [])

    # Preparing the data frame for our train data
    columns = ['ft_5', 'ft_4', 'ft_3', 'ft_2', 'ft_1']
    df_train = pd.DataFrame(data=train_new_features, columns=columns)
    df_train['lat'] = train_lat
    df_train['lon'] = train_lon
    df_train['weekday'] = train_weekday
    df_train['exp_avg'] = train_exp_avg

    # Preparing the data frame for our train data
    df_test = pd.DataFrame(data=test_new_features, columns=columns)
    df_test['lat'] = test_lat
    df_test['lon'] = test_lon
    df_test['weekday'] = test_weekday
    df_test['exp_avg'] = test_exp_avg

    df_test_lm = pd.concat([df_test, final_test_df], axis=1)
    df_train_lm = pd.concat([df_train, final_train_df], axis=1)
    # df_train_lm=df_train_lm.isnull().fillna(0)
    # print(df_test_lm.head())
    # print(df_train_lm.head())
    # print(df_test_lm.columns)
    # print(df_train_lm.columns)

    # nan_rows = df_train_lm[df_train_lm.isnull().T.any().T]
    # print(nan_rows)
    # print(df_train_lm.head(3))
    return df_train_lm, df_test_lm, train_output


def split():
    # aug 31- 31*24*60/10 = 4464 time bins - 10 min
    # oct 31 - 31*24*60/10 = 4464 time bins - 10 min
    # Nov 30 -  30*24*60/10 = 4320 time bins - 10 min
    regions_cum = []

    df_2013_08 = pd.read_csv("Datasets/preprocess_2013_08.csv")
    df_2014_08 = pd.read_csv("Datasets/preprocess_2014_08.csv")
    df_2014_09 = pd.read_csv("Datasets/preprocess_2014_09.csv")
    df_2014_10 = pd.read_csv("Datasets/preprocess_2014_10.csv")

    # df_unique_2013_08 = get_unique_bins(df_2013_08)
    df_unique_2014_08 = get_unique_bins(df_2014_08)
    df_unique_2014_09 = get_unique_bins(df_2014_09)
    df_unique_2014_10 = get_unique_bins(df_2014_10)

    # group_2013_08 = df_2013_08[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
    # ['pickup_cluster', 'pickup_bins']).count()
    group_2014_08 = df_2014_08[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
        ['pickup_cluster', 'pickup_bins']).count()
    group_2014_09 = df_2014_09[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
        ['pickup_cluster', 'pickup_bins']).count()
    group_2014_10 = df_2014_10[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
        ['pickup_cluster', 'pickup_bins']).count()

    # smooth_2013_08 = smooth(group_2013_08['trip_distance'].values, df_unique_2013_08)
    smooth_2014_08 = fill_zeros_bin(group_2014_08['trip_distance'].values, df_unique_2014_08)
    smooth_2014_09 = fill_zeros_bin(group_2014_09['trip_distance'].values, df_unique_2014_09)
    smooth_2014_10 = fill_zeros_bin(group_2014_10['trip_distance'].values, df_unique_2014_10)

    for i in range(0, 30):
        regions_cum.append(
            smooth_2014_08[4464 * i:4464 * (i + 1)] + smooth_2014_09[4464 * i:4464 * (i + 1)] + smooth_2014_10[4320 * i:
                                                                                                               4320 * (
                                                                                                                       i + 1)])
    coords = df_2013_08[['pickup_latitude', 'pickup_longitude']].values
    kmeans = MiniBatchKMeans(n_clusters=30, batch_size=10000, random_state=42).fit(coords)
    train, test, output = prepare_for_split(regions_cum, kmeans)
    linear_regression(train, test, output)


def linear_regression(train, test, train_output):
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

    y_pred = clf.predict(std_test)
    lr_test_predictions = [round(value) for value in y_pred]
    y_pred = clf.predict(std_train)
    lr_train_predictions = [round(value) for value in y_pred]


def random_forest_regression(train, test, train_output):
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

    y_pred = clf1.predict(test)
    rndf_test_predictions = [round(value) for value in y_pred]
    y_pred = clf1.predict(train)
    rndf_train_predictions = [round(value) for value in y_pred]


if __name__ == '__main__':
    split()
