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
    
    # Handle categorical variables - encode employment_status
    if 'employment_status' in df.columns:
        df['employment_status_employed'] = (df['employment_status'] == 'employed').astype(int)
        df['employment_status_self_employed'] = (df['employment_status'] == 'self-employed').astype(int)
        df['employment_status_unemployed'] = (df['employment_status'] == 'unemployed').astype(int)
        df = df.drop('employment_status', axis=1)
    
    # Handle loan_type encoding
    if 'loan_type' in df.columns:
        df['loan_type_cash'] = (df['loan_type'] == 'cash_loan').astype(int)
        df['loan_type_revolving'] = (df['loan_type'] == 'revolving_loan').astype(int)
        df = df.drop('loan_type', axis=1)
    
    # Split features/target (assume 'default' is target)
    X = df.drop('default', axis=1)
    y = df['default']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to preserve column names
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    
    return X_train_df, X_test_df, y_train, y_test 