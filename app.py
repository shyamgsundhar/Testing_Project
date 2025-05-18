from flask import Flask, request, jsonify
import pickle
import numpy as np
import os  # <-- import os to get PORT

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

# THIS IS THE IMPORTANT PART
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # use the PORT from environment
    app.run(host='0.0.0.0', port=port)
