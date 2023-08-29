from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

app = Flask(__name__)

loaded_lstm_model = load_model("lstm_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

with open("scaler.pkl", "rb") as f:
    loaded_scaler = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract features from the incoming data
    features_from_user = np.array([
        float(data['sex']), float(data['painexer']), float(data['cp']), float(data['smoke']),
        float(data['fbs']), float(data['dm']), float(data['prop']), float(data['nitr']),
        float(data['pro']), float(data['proto']), float(data['thalach']), float(data['exang']),
        float(data['slope']), float(data['ca'])
    ])

    features_preset = np.zeros(33)

    combined_features = np.concatenate([features_from_user, features_preset])

    combined_features = combined_features.reshape(1, -1)

    combined_features_scaled = loaded_scaler.transform(combined_features)

    selected_features_scaled = combined_features_scaled[:, :14]

    selected_features_scaled = selected_features_scaled.reshape(1, 14, 1)
    prediction = loaded_lstm_model.predict(selected_features_scaled)

    # Interpret the result
    if prediction[0][0] >= 0.5:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease Detected"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)