import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    Loads the Parkinson's dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None

def preprocess_data(df):
    """
    Cleans the data, separates features/target, and scales the features.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    # 1. Drop the 'name' column (identifier, not a feature)
    if 'name' in df.columns:
        df = df.drop(['name'], axis=1)

    # 2. Separate Features (X) and Target (y)
    # The target column in this dataset is 'status'
    X = df.drop(['status'], axis=1)
    y = df['status']

    # 3. Split into Train and Test sets (80% Train, 20% Test)
    # random_state=42 ensures the split is the same every time
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Scale the features (MinMax Scaling: -1 to 1)
    # This is crucial for algorithms like SVM and KNN
    scaler = MinMaxScaler((-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler