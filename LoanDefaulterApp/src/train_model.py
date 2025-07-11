import joblib
from sklearn.ensemble import RandomForestClassifier
from src.data_preprocessing import load_and_preprocess_data

def train_and_save_model(data_path, model_path):
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model, X_test, y_test 