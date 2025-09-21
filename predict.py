import joblib
import numpy as np

def load_model(path="../models/model.pkl"):
    return joblib.load(path)

def predict(input_data):
    bundle = load_model()
    model = bundle["model"]
    scaler = bundle["scaler"]

    input_data = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    return prediction[0]

if __name__ == "__main__":
    sample_input = [70, 80, 8.5]  # Example
    result = predict(sample_input)
    print("Prediction:", result)
