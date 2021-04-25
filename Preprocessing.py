import pandas as pd
import requests
from sklearn.cluster import KMeans
import math
import numpy as np
import pickle
from sklearn.ensemble import IsolationForest

import warnings

warnings.simplefilter('ignore')


def preprocess_dataset(filepath):
    df = pd.read_csv(filepath)
    df.drop(['vendor_id', 'rate_code', 'store_and_fwd_flag', 'tip_amount', 'payment_type', 'fare_amount', 'surcharge',
             'mta_tax',
             'tolls_amount', 'total_amount'], axis='columns',
            inplace=True)
    # Drop null values in dataset
    df = df.dropna(subset=['dropoff_longitude', 'dropoff_latitude'])

    # Filter pickup locations and drop locations within manhattan
    manhattan_df = df[(df.pickup_latitude > 40.7091) &
                      (df.pickup_latitude < 40.8205) &
                      (df.pickup_longitude > -74.0096) &
                      (df.pickup_longitude < -73.9307) &
                      (df.dropoff_latitude > 40.7091) &
                      (df.dropoff_latitude < 40.8205) &
                      (df.dropoff_longitude > -74.0096) &
                      (df.dropoff_longitude < -73.9307)
                      ]

    manhattan_df['pickup_datetime'] = pd.to_datetime(manhattan_df['pickup_datetime'])
    manhattan_df['dropoff_datetime'] = pd.to_datetime(manhattan_df['dropoff_datetime'])
    # manhattan_df['hour_of_day'] = manhattan_df['pickup_datetime'].dt.hour
    manhattan_df['week_day'] = manhattan_df['pickup_datetime'].dt.weekday

    manhattan_df['trip_duration'] = (
            manhattan_df['dropoff_datetime'] - manhattan_df['pickup_datetime']).dt.total_seconds()

    manhattan_df['speed'] = (manhattan_df['trip_distance'] / manhattan_df['trip_duration']) * 60

    # manhattan_df['bin'] = (manhattan_df['pickup_datetime'].sub(manhattan_df['pickup_datetime'].min())
    #                        .dt.floor('10Min')
    #                        .rank(method='dense')
    #                        .astype(int))

    manhattan_df['bin'] = math.ceil(df['pickup_datetime'].minute / 5)

    # clustering pickup points and drop points
    kmeans_p = KMeans(n_clusters=30).fit(manhattan_df[['pickup_latitude', 'pickup_longitude']])
    # kmeans_d = KMeans(n_clusters=30).fit(manhattan_df[['dropoff_latitude', 'dropoff_longitude']])
    manhattan_df['pickup_cluster'] = kmeans_p.predict(manhattan_df[['pickup_latitude', 'pickup_longitude']])
    manhattan_df['dropoff_cluster'] = kmeans_p.predict(manhattan_df[['dropoff_latitude', 'dropoff_longitude']])
    # manhattan_df['clusters'] = manhattan_df['pickup_cluster'].map(str) + manhattan_df['dropoff_cluster'].map(str)
    # manhattan_df.to_csv('Datasets/preprocess_08_2013.csv')
    filtered_df_2013_08 = filter_outliers(manhattan_df)
    # print(filtered_df_2013_08.columns)
    filtered_df_2013_08.to_csv('filtered_df_2013_08.csv')


def filter_outliers(df):
    # remove outliers based on trip duration
    duration_filter_df = df[(df.trip_duration > 0) & (df.trip_duration < 720)]
    print('duration df: ', len(duration_filter_df))

    # # remove outlier based on speed
    speed_filter_df = duration_filter_df[(duration_filter_df.speed > 0) & (duration_filter_df.speed < 48.6)]
    print('Speed df: ', len(speed_filter_df))

    # # remove outliers based on distance
    distance_filter_df = speed_filter_df[(speed_filter_df.trip_distance > 0) & (speed_filter_df.trip_distance < 22.25)]
    print('Distance df: ', len(distance_filter_df))

    # Filter passenger outliers
    passenger_count_filter_df = distance_filter_df[speed_filter_df.passenger_count <= 6]
    print('Passenger Count df: ', len(passenger_count_filter_df))

    return passenger_count_filter_df


# make osrm api get call to get distance
def get_route_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    # call the OSMR API
    loc = "{},{};{},{}".format(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    url = "http://127.0.0.1:5000/route/v1/car/"

    r = requests.get(url + loc + "?steps=true", verify=False, timeout=10)
    if r.status_code != 200:
        return 0.0, 0.0, 0, 0
    res = r.json()
    route_distance = res.get("routes")[0]['distance']
    route_duration = res.get("routes")[0]['duration']
    steps = len(res.get("routes")[0]['legs'][0]['steps'])
    intersections = len(res.get("routes")[0]['legs'][0]['steps'][0]['intersections'])
    return route_distance, route_duration, steps, intersections


# append osrm distance to dataframe
def get_distance(df):
    distanceArray = []
    durationArray = []
    stepsArray = []
    intersectionsArray = []
    for ind in df.index:
        distance, duration, steps, intersections = get_route_distance(df['pickup_longitude'][ind],
                                                                      df['pickup_latitude'][ind],
                                                                      df['dropoff_longitude'][ind],
                                                                      df['dropoff_latitude'][ind])
        distanceArray.append(distance)
        durationArray.append(duration)
        stepsArray.append(steps)
        intersectionsArray.append(intersections)
        print(len(distanceArray))
    df['osrm_distance'] = distanceArray
    df['osrm_duration'] = durationArray
    df['steps'] = stepsArray
    df['intersections'] = intersectionsArray

    return df


def preprocess(filepath, month, year, name):
    # Step 1
    dataFrame = preprocess_dataset(filepath)
    # Step 2
    # cluster_df= create_clusters(dataFrame)

    # pickup_df = create_pickup_bins(cluster_df, month, year)

    # pickup_df.to_csv(name)


def iqr_bounds(data, iqr_threshold=1.5, verbose=False):
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q1 - q3

    lower_bound = q1 - iqr_threshold * iqr
    upper_bound = q3 + iqr_threshold * iqr

    return lower_bound, upper_bound


def outlier_detection_using_isolation_forest(df):
    iso = IsolationForest(contamination=0.3)
    iso.fit(df['trip_distance'].values.reshape(-1, 1))
    df['anomaly_score_trip_distance'] = iso.decision_function(df['trip_distance'].values.reshape(-1, 1))
    df['anomaly_trip_distance'] = iso.predict(df['trip_distance'].values.reshape(-1, 1))

    df['anomaly_trip_duration'] = iso.fit_predict(df['trip_duration'])

    print(len(df['anomaly_trip_distance'] == -1))
    print(len(df['anomaly_trip_distance'] == 1))

    print(len(df['anomaly_trip_distance'] == -1))
    print(len(df['anomaly_trip_distance'] == 1))



if __name__ == '__main__':
    # pd.set_option('display.width', 320)

    # np.set_printoptions(linewidth=320)

    # pd.set_option('display.max_columns', 30)

    # df = pd.read_csv('Datasets/preprocess_08_2013.csv')
    # filter_df = filter_outliers(df)
    # df = pd.read_csv('filtered_df_2013_08.csv')
    # print(df.columns)
    # osrm_df = get_distance(df)
    # osrm_df.to_csv('Datasets/osrm_08_2013.csv')

    df1 = pd.read_csv('Datasets/osrm_08_2013.csv')
    df1.drop(['Unnamed: 0', 'Unnamed: 0.1', 'passenger_count', 'speed', 'clusters'], axis='columns',
             inplace=True)

    # remove outliers based on distance and duration ratio
    df1['distance_ratio'] = (df1['trip_distance'] * 1.60934) / df1['osrm_distance']
    df1['duration_ratio'] = df1['trip_duration'] / df1['osrm_duration']
    lower_dist, upper_dist = iqr_bounds(df1['distance_ratio'])
    lower_dura, upper_dura = iqr_bounds(df1['duration_ratio'])
    df1 = df1[(df1.distance_ratio >= upper_dist) & (df1.duration_ratio >= upper_dura)]

    # remove infinte and nan values
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1.dropna(inplace=True)
    df1.drop(['bin'], axis='columns', inplace=True)
    df1['pickup_datetime'] = pd.to_datetime(df1['pickup_datetime'])
    # df1['bin'] = math.ceil(df1['pickup_datetime'].dt.minute / 10)
    df1['bin'] = df1['pickup_datetime'].apply(lambda x: math.ceil(x.minute / 10))
    df1.drop(['distance_ratio', 'duration_ratio', 'pickup_datetime', 'dropoff_datetime',
              'trip_distance'], axis='columns', inplace=True)
    df1.to_csv('Datasets/preprocessed.csv')

    ## outlier_detection_using_isolation_forest(df1)
    kmeans_p = KMeans(n_clusters=30).fit(df1[['pickup_latitude', 'pickup_longitude']])
    pickle.dump(kmeans_p, open("kmeans.pkl", "wb"))
