import joblib
import numpy as np

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_cancer(features):

    features = np.array(features).reshape(1, -1)

    features = scaler.transform(features)

    prediction = model.predict(features)

    if prediction == 1:
        return "Benign"
    else:
        return "Malignant"
