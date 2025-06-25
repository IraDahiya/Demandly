import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_features(df):
    # Convert 'date' to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract day of week and month as new features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    return df

def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def load_and_preprocess_data(filepath, seq_length=7):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Feature engineering
    df = create_features(df)
    
    # Use demand and engineered features as input features
    # We'll scale numeric features after sequence creation
    features = ['demand', 'day_of_week', 'month']
    data = df[features].values.astype(float)
    
    # Create sequences
    X, y = create_sequences(data, seq_length)
    
    # Split data into train and test sets (e.g., last 20% for testing)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features using MinMaxScaler for each feature dimension
    scaler_X = MinMaxScaler()
    # Reshape to 2D for scaler: (samples*seq_length, features)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_train = X_train_scaled.reshape(X_train.shape)
    
    # Scale test set
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
    X_test_scaled = scaler_X.transform(X_test_reshaped)
    X_test = X_test_scaled.reshape(X_test.shape)
    
    # Scale target variable (only demand, which is y[:, 0])
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train[:, 0].reshape(-1,1))
    y_test_scaled = scaler_y.transform(y_test[:, 0].reshape(-1,1))
    
    return X_train, X_test, y_train_scaled, y_test_scaled, scaler_X, scaler_y
