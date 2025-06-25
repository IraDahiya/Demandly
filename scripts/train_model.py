import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt

def create_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    return df

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq_x = data[i:i+seq_length]
        seq_y = data[i+seq_length, 0]  # assuming first column is target (demand)
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X)
    y = np.array(y)
    print("Created sequences shape:", X.shape, y.shape)
    return X, y


def load_and_preprocess_data(filepath, seq_length=7):
    df = pd.read_csv(filepath)
    print("Dataset rows:", len(df))
    
    df = create_features(df)
    features = ['demand', 'day_of_week', 'month']
    data = df[features].values.astype(float)
    print("Data shape:", data.shape)
    
    X, y = create_sequences(data, seq_length)
    print("Sequences created - X shape:", X.shape, "y shape:", y.shape)
    
    if X.size == 0:
        raise ValueError(f"No sequences created. Dataset too small for seq_length={seq_length}. Reduce seq_length or add more data.")
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler_X = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_scaled = scaler_X.fit_transform(X_train_reshaped)
    X_train = X_train_scaled.reshape(X_train.shape)
    
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler_X.transform(X_test_reshaped)
    X_test = X_test_scaled.reshape(X_test.shape)
    
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    return X_train, X_test, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


if __name__ == "__main__":
    filepath = r"C:\Users\irada\Demandly\raw_data.csv"  # your dataset path
    seq_length = 3  # number of days in input sequence
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess_data(filepath, seq_length)
    
    # Build model
    model = build_lstm_model(input_shape=(seq_length, X_train.shape[2]))
    
    # Train model
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=16, callbacks=[early_stop], verbose=1)
    
    # Predict on test data
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform to get original demand values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_test)
    
    # Evaluate
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label='Actual Demand')
    plt.plot(y_pred, label='Predicted Demand')
    plt.title('Actual vs Predicted Demand')
    plt.xlabel('Test Data Points')
    plt.ylabel('Demand')
    plt.legend()
    plt.show()

   