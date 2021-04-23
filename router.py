from flask import Flask, request

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form.get('text')



if __name__ == '__main__':
    app.run(debug=True, port=5050)
