import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
import sys

# Add the current directory to the system path so we can import 'preprocessing'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our preprocessing function from preprocessing.py
from preprocessing import load_data, preprocess_data

def train_model():
    # 1. Load Data
    # Adjust this path if your file is named differently
    data_path = os.path.join('data', 'raw', 'parkinsons.data')
    
    print(f"Loading data from: {data_path}...")
    df = load_data(data_path)

    if df is not None:
        # 2. Preprocess Data
        # We get the scaled features and the scaler itself
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

        # 3. Initialize the Model (XGBoost Classifier)
        # We can tune hyperparameters here, but defaults work well for a start
        model = XGBClassifier(eval_metric='logloss')

        # 4. Train the Model
        print("Training the model...")
        model.fit(X_train, y_train)

        # 5. Evaluate the Model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # 6. Save the Model and Scaler
        # We MUST save the scaler too, or new data won't match the model's expected range
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    train_model()