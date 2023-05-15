from flask import Flask, request, jsonify
from joblib import load
from flask_cors import CORS # import CORS

model = load('./models/model.joblib')

app = Flask(__name__)
CORS(app) # enable CORS for your app

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']

    prediction = model.predict(data)
    feature_importances = model.feature_importances_.tolist()
    node_impurities = model.tree_.impurity.tolist()

    response = {
        'prediction': prediction.tolist(),
        'feature_importances': feature_importances,
        'node_impurities': node_impurities
    }

    return jsonify(response)

@app.route('/')
def index():
    return 'Hello from Flask!'

if __name__ == '__main__':
    app.run()
