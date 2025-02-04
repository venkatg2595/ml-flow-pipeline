from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load Model
model = joblib.load("model/model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features).tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
