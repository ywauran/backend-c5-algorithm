from flask import Flask, request, jsonify
from joblib import load

model = load('./models/model.joblib')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']

    prediction = model.predict(data)

    return jsonify({'prediction': prediction.tolist()})

@app.route('/')
def index():
    return 'Hello from Flask!'

if __name__ == '__main__':
    app.run()
