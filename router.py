from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from datetime import datetime
from Preprocessing import get_route_distance
import math
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/hello', methods=['GET'])
def hello():
    return "hello", 200


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    data = request.get_json(force=True)
    pickup_lat = data['pickup_lat']
    pickup_lon = data['pickup_lon']
    drop_lat = data['drop_lat']
    drop_lon = data['drop_lon']
    date_time_str = data['timestamp']
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
    bin = math.ceil(date_time_obj.minute / 10)
    dist,dura, steps, intersections = get_route_distance(pickup_lon, pickup_lat, drop_lon, drop_lat)
    file = open("kmeans.pkl", 'rb')
    kmeans = pickle.load(file)

    data = {'pickup_longitude': [pickup_lon],
            'pickup_latitude': [pickup_lat],
            'dropoff_longitude': [drop_lon],
            'dropoff_latitude': [drop_lat],
            'hour_of_day': [date_time_obj.hour],
            'week_day': [date_time_obj.weekday()],
            'pickup_cluster': [1],
            'dropoff_cluster': [1],
            'osrm_distance': [dist],
            'osrm_duration': [dura],
            'steps': [steps],
            'intersections': [intersections],
            'bin': [bin],
            }

    df = pd.DataFrame(data)
    df['pickup_cluster'] = kmeans.predict(df[['pickup_latitude', 'pickup_longitude']])
    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_latitude', 'dropoff_longitude']])
    file1 = open("Models/stacker.pkl", 'rb')
    cat = pickle.load(file1)
    predict = cat.predict(df)
    result = {'result': predict[0]}
    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True, port=5050)
