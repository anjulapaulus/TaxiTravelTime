import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from datetime import datetime
import time
import matplotlib
import datetime
import requests
from sklearn.cluster import MiniBatchKMeans

matplotlib.use('nbagg')
import warnings

warnings.simplefilter('ignore')

thread_local = threading.local()

distanceArray = []


def preprocess_dataset(filepath):
    df = pd.read_csv(filepath)

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
    # convert date into unix timestamp
    time_df = time_duration_data_frame(manhattan_df)

    # remove outliers based on trip duration
    duration_filter_df = time_df[(time_df.trip_duration > 0) & (time_df.trip_duration < 720)]

    # # remove outlier based on speed
    speed_filter_df = duration_filter_df[(duration_filter_df.Speed > 0) & (duration_filter_df.Speed < 48.6)]

    # # remove outliers based on distance
    distance_filter_df = speed_filter_df[(speed_filter_df.trip_distance > 0) & (speed_filter_df.trip_distance < 22.25)]

    return distance_filter_df


def preprocess_dataset_osrm(filepath):
    df = pd.read_csv(filepath)
    df = get_distance(df)
    df.to_csv("Datasets/distance_df.csv")


def convert_datetime_to_unix(s):
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())


# convert datetime to unix timestamp - seconds
def time_duration_data_frame(month):
    # pickups and dropoffs to unix time
    pickup_duration = [convert_datetime_to_unix(x) for x in month['pickup_datetime'].values]
    drop_duration = [convert_datetime_to_unix(x) for x in month['dropoff_datetime'].values]
    # calculate duration of trips
    durations = (np.array(drop_duration) - np.array(pickup_duration)) / float(60)

    # append durations of trips and speed in miles/hr to a new dataframe
    new_df = month[['passenger_count', 'trip_distance',
                    'pickup_longitude', 'pickup_latitude',
                    'dropoff_longitude', 'dropoff_latitude']]

    new_df['trip_duration'] = durations
    new_df['pickup_duration'] = pickup_duration
    new_df['Speed'] = 60 * (new_df['trip_distance'] / new_df['trip_duration'])

    return new_df


# make osrm api get call to get distance
def get_route_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat):
    # call the OSMR API
    loc = "{},{};{},{}".format(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    url = "http://127.0.0.1:5000/route/v1/car/"

    r = requests.get(url + loc)
    if r.status_code != 200:
        return 0
    res = r.json()
    route_distance = res.get("routes")[0]['distance']
    route_duration = res.get("routes")[0]['duration']
    return route_distance, route_duration


# append osrm distance to dataframe
def get_distance(df):
    threads = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        for ind in df.index:
            threads.append(executor.submit(get_route_distance, df['pickup_longitude'][ind], df['pickup_latitude'][ind],
                                           df['dropoff_longitude'][ind], df['dropoff_latitude'][ind]))
            # distance = get_route_distance(df['pickup_longitude'][ind], df['pickup_latitude'][ind],
            #                               df['dropoff_longitude'][ind], df['dropoff_latitude'][ind])
            # distanceArray.insert(ind, distance)
        for task in as_completed(threads):
            distanceArray.append(task.result())
            print(len(distanceArray))
    df['osrm_distance'] = distanceArray
    return df


clusters = 30


def create_clusters(df):
    # The inter cluster distance should be minimum of 0.5 miles.
    coordinates = df[['pickup_latitude',
                      'pickup_longitude']].values
    # Getting 30 clusters using the kmeans
    kmeans = MiniBatchKMeans(n_clusters=clusters, batch_size=10000, random_state=0).fit(coordinates)
    df['pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    return df


def create_pickup_bins(df, month, year):
    # 1356978600 - 2013-01-01 00.00.00
    # 1359657000 - 2013-02-01 00.00.00
    # 1362076200 - 2013-03-01 00.00.00
    # 1364754600 - 2013-04-01 00.00.00
    # 1367346600 - 2013-05-01 00.00.00
    # 1370025000 - 2013-06-01 00.00.00
    # 1372617000 - 2013-07-01 00.00.00
    # 1375295400 - 2013-08-01 00.00.00
    # 1377973800 - 2013-09-01 00.00.00
    # 1380565800 - 2013-10-01 00.00.00
    # 1383244200 - 2013-11-01 00.00.00
    # 1385836200 - 2013-12-01 00.00.00

    # 1388514600 - 2014-01-01 00.00.00
    # 1391193000 - 2014-02-01 00.00.00
    # 1393612200 - 2014-03-01 00.00.00
    # 1396290600 - 2014-04-01 00.00.00
    # 1398882600 - 2014-05-01 00.00.00
    # 1401561000 - 2014-06-01 00.00.00
    # 1404153000 - 2014-07-01 00.00.00
    # 1406831400 - 2014-08-01 00.00.00
    # 1409509800 - 2014-09-01 00.00.00
    # 1412101800 - 2014-10-01 00.00.00
    # 1414780200 - 2014-11-01 00.00.00
    # 1417372200 - 2014-12-01 00.00.00

    unix_pickup_times = [i for i in df['pickup_duration'].values]
    unix_times = [
        [1356978600, 1359657000, 1362076200, 1364754600, 1367346600, 1370025000, 1372617000, 1375295400, 1377973800,
         1380565800, 1383244200, 1385836200],
        [1388514600, 1391193000, 1393612200, 1396290600, 1398882600, 1401561000, 1404153000, 1406831400, 1409509800,
         1412101800, 1414780200, 1417372200]]

    start_pickup_unix = unix_times[year - 2013][month - 1]
    pickup_times = [(int((i - start_pickup_unix) / 600) + 33) for i in unix_pickup_times]
    df['pickup_bins'] = np.array(pickup_times)
    return df


# get unique time bins where pickups are present for each cluster
def get_unique_bins(dataframe):
    values = []
    for i in range(0, clusters):
        new = dataframe[dataframe['pickup_cluster'] == i]
        list_unqiue = list(set(new['pickup_bins']))
        list_unqiue.sort()
        values.append(list_unqiue)
    return values


num_of_10_min_bins = int(31 * 24 * (60 / 10))


# fill zeros for bins where pickups are not present
def fill_zeros_bin(count, val):
    smooth_region = []
    index = 0
    for r in range(0, clusters):
        smooth_bins = []
        for i in range(num_of_10_min_bins):
            if i in val[r]:
                smooth_bins.append(count[index])
                index += 1
            else:
                smooth_bins.append(0)
        smooth_region.extend(smooth_bins)
    return smooth_region

def smooth(count, val):
    regions_smooth = []
    index = 0
    repeat = 0
    smooth_val = 0
    for r in range(0, clusters):
        bins_smooth = []
        repeat = 0
        for i in range(num_of_10_min_bins):
            if repeat != 0:
                repeat -= 1
                continue
            if i in val[r]:
                bins_smooth.append(count[index])
            else:
                if i != 0:
                    rhand_limit = 0
                    for j in range(i, num_of_10_min_bins):
                        if j not in val[r]:
                            continue
                        else:
                            rhand_limit = j
                            break
                    if rhand_limit == 0:
                        smooth_val = count[index - 1] * 1.0 / (((num_of_10_min_bins - 1) - i) + 2) * 1.0
                        for j in range(i, 4464):
                            bins_smooth.append(math.ceil(smooth_val))
                        bins_smooth[i - 1] = math.ceil(smooth_val)
                        repeat = (4463 - i)
                        index -= 1
                    else:
                        smooth_val = (count[index - 1] + count[index]) * 1.0 / ((rhand_limit - i) + 2) * 1.0
                        for j in range(i, rhand_limit + 1):
                            bins_smooth.append(math.ceil(smooth_val))
                        bins_smooth[i - 1] = math.ceil(smooth_val)
                        repeat = (rhand_limit - i)
                else:
                    rhand_limit = 0
                    for j in range(i, 4464):
                        if j not in val[r]:
                            continue
                        else:
                            rhand_limit = j
                            break
                    smooth_val = count[index] * 1.0 / ((rhand_limit - i) + 1) * 1.0
                    for j in range(i, rhand_limit + 1):
                        bins_smooth.append(math.ceil(smooth_val))
                    repeat = (rhand_limit - i)
            index += 1
        regions_smooth.extend(bins_smooth)
    return regions_smooth


def smooth_pickup_bins_dataset(dataframe):
    df_unique = get_unique_bins(dataframe)
    df_group = dataframe[['pickup_cluster', 'pickup_bins', 'trip_distance']].groupby(
        ['pickup_cluster', 'pickup_bins']).count()

    # df_fill = fill_zeros_bin(df_group['trip_distance'].values, df_unique)

    df_smooth = smooth(df_group['trip_distance'].values, df_unique)

    return df_smooth


def preprocess(filepath, month, year, name):
    # Step 1
    dataFrame = preprocess_dataset(filepath)
    # Step 2
    cluster_df= create_clusters(dataFrame)

    pickup_df = create_pickup_bins(cluster_df, month, year)

    pickup_df.to_csv(name)


if __name__ == '__main__':
    preprocess("Datasets/yellow_tripdata_2013-08.csv", 8,  2013,  "Datasets/preprocess_2013_08.csv")
    preprocess("Datasets/yellow_tripdata_2014-08.csv", 8,  2014,  "Datasets/preprocess_2014_08.csv")
    preprocess("Datasets/yellow_tripdata_2014-09.csv", 9,  2014,  "Datasets/preprocess_2014_09.csv")
    preprocess("Datasets/yellow_tripdata_2014-10.csv", 10, 2014,  "Datasets/preprocess_2014_10.csv")
