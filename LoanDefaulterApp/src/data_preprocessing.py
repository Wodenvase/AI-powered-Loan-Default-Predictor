import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(filepath):
    """
    Loads data from a CSV file, handles missing values, encodes categoricals, and splits into train/test sets.
    """
    df = pd.read_csv(filepath)
    # Example preprocessing: fill missing values
    df = df.fillna(df.median(numeric_only=True))
    # Encode categoricals (simple example)
    df = pd.get_dummies(df)
    # Split features/target (assume 'default' is target)
    X = df.drop('default', axis=1)
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test 