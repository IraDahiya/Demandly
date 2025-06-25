import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from preprocessing import load_and_preprocess_data

# Absolute path to your dataset CSV file
filepath = r"C:\Users\irada\Demandly\raw_data.csv"

# Load data and scalers
X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_preprocess_data(filepath)

# Build simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=1,
    validation_data=(X_test, y_test),
    verbose=1
)

# Evaluate on test data
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {test_mae}")

# Predict on test data
predictions = model.predict(X_test)
print("Predictions (scaled):", predictions)

# Inverse transform predictions and actual y_test
y_pred_rescaled = scaler_y.inverse_transform(predictions)
y_test_rescaled = scaler_y.inverse_transform(y_test)

print("Predictions (original scale):", y_pred_rescaled)
print("Actual values (original scale):", y_test_rescaled)
