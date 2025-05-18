from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Linear Regression Model is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_features = np.array(data['input']).reshape(1, -1)
    prediction = model.predict(input_features)
    return jsonify({'prediction': prediction.tolist()})
