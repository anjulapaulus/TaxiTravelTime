from flask import Flask, request
from datetime import datetime

app = Flask(__name__)


@app.route('/hello', methods=['GET'])
def hello():
    return "hello", 200


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pickup_lat = data['pickup_lat']
    pickup_lon = data['pickup_lon']
    drop_lat = data['drop_lat']
    drop_lon = data['drop_lon']
    timestamp = data['timestamp']
    pickup_datetime = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
    data = {'pickup_longitude': [pickup_lon],
            'pickup_latitude': [pickup_lat],
            'dropoff_longitude': [drop_lon],
            'dropoff_latitude': [drop_lat],
            'week_day': ['First value', 'Second value', ...],
            'bin': ['First value', 'Second value', ...],
            'pickup_cluster': ['First value', 'Second value', ...],
            'dropoff_cluster': ['First value', 'Second value', ...],
            'osrm_distance': ['First value', 'Second value', ...],
            'osrm_duration': ['First value', 'Second value', ...],
            'steps': ['First value', 'Second value', ...],
            'intersections': ['First value', 'Second value', ...],
            }

    print(pickup_lat)
    return "okay", 200


if __name__ == '__main__':
    app.run(debug=True, port=5050)
