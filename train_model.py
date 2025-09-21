import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_data, preprocess_data

def train_and_save_model():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    joblib.dump({"model": model, "scaler": scaler}, "../models/model.pkl")
    print("✅ Model saved to ../models/model.pkl")

if __name__ == "__main__":
    train_and_save_model()
