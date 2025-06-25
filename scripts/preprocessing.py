import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear

    X = df[['day_of_year']].values
    y = df['demand'].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_X, scaler_y

def train_and_predict(X_scaled, y_scaled, scaler_X, scaler_y):
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)

    # Predict demand for day 366 (next year)
    next_day = [[366]]
    next_day_scaled = scaler_X.transform(next_day)
    pred_scaled = model.predict(next_day_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)
    print(f"Predicted demand for day {next_day[0][0]}: {pred[0][0]:.2f}")

if __name__ == "__main__":
    filepath = r"C:\Users\irada\Demandly\raw_data.csv"
    X_scaled, y_scaled, scaler_X, scaler_y = load_and_preprocess_data(filepath)
    train_and_predict(X_scaled, y_scaled, scaler_X, scaler_y)

