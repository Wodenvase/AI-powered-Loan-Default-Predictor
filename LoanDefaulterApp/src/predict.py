import joblib
import numpy as np

def load_model(model_path):
    return joblib.load(model_path)

def predict_default(model, input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)
    return prediction[0], probability[0][1] 