import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# --- Configuration ---
STOCK_TICKER = 'AAPL'  # Apple Inc. - Common example
START_DATE = '2023-01-01'
END_DATE = '2025-12-08'
SEQUENCE_LENGTH = 60  # Number of previous days used to predict the next day (T-60 to T-1 to predict T)
FUTURE_DAYS = 750  # Approximately 3 years of trading days


def load_data(ticker, start, end):
    """Downloads historical stock data using yfinance."""
    print(f"Loading data for {ticker} from {start} to {end}...")
    # auto_adjust=True fetches adjusted closing prices directly
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if df.empty:
        raise ValueError("Failed to load data. Check ticker and date range.")
    return df


def preprocess_data(df, sequence_length):
    """
    Scales the data and creates time-series sequences for the entire dataset.
    Also extracts the last sequence needed to start future prediction.
    """
    # Use only the 'Close' price for prediction (univariate time series)
    data = df['Close'].values.reshape(-1, 1)

    # 1. Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # 2. Create the sequence data (X_all and y_all)
    X_all, y_all = [], []
    for i in range(sequence_length, len(scaled_data)):
        # X gets the past 'sequence_length' days
        X_all.append(scaled_data[i - sequence_length:i, 0])
        # y gets the price of the current day
        y_all.append(scaled_data[i, 0])

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # 3. Reshape X for LSTM input (samples, time steps, features)
    X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1], 1))

    # 4. Get the last known sequence to start the future forecast
    last_known_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)

    print(f"Total sequences created: {len(X_all)}")
    print(f"LSTM Input Shape (Full Data): {X_all.shape}")

    # Return X_all, y_all (full training set), scaler, full df, and the starting sequence
    return X_all, y_all, scaler, df, last_known_sequence


def build_lstm_model(input_shape):
    """
    Defines the LSTM neural network architecture. [Image of LSTM model architecture layers]
    """
    model = Sequential()

    # Layer 1: LSTM with 50 units. return_sequences=True
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Layer 2: return_sequences=False
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Dense layer for feature reduction
    model.add(Dense(units=25))

    # Output layer: 1 unit for the predicted 'Close' price
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("\n--- Model Summary ---")
    model.summary()
    return model


def predict_future_prices(model, last_sequence, scaler, future_days):
    """
    Performs multi-step iterative forecasting for 'future_days'.
    The model's output is fed back as the input for the next day's prediction.
    """
    future_predictions = []
    current_sequence = last_sequence.copy()  # Use a copy to avoid modifying the original

    print(f"\n--- Starting {future_days}-Day Iterative Forecast ---")

    for _ in range(future_days):
        # 1. Predict the next day's scaled price
        predicted_scaled_price = model.predict(current_sequence, verbose=0)[0, 0]

        # 2. Store the prediction
        future_predictions.append(predicted_scaled_price)

        # 3. Update the sequence for the next day's prediction (Iterative step)
        # Drop the oldest price (index 0) and append the new prediction at the end.

        # current_sequence shape: (1, 60, 1)
        # Extract the 60 steps, remove the first one, and add the prediction
        new_sequence = np.append(current_sequence[:, 1:, :], np.array([[[predicted_scaled_price]]]), axis=1)

        # Update the sequence for the next iteration
        current_sequence = new_sequence

    # Convert the scaled predictions back to actual USD prices
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions.flatten()


def plot_results(df, future_predictions, sequence_length, future_days):
    """
    Plots the historical data and the future forecast.
    """
    # 1. Prepare future dates
    last_date = df.index[-1]

    # Generate subsequent business days for the prediction period
    future_dates = pd.date_range(start=last_date, periods=future_days + 1, freq='B')[1:]

    # 2. Create Future DataFrame
    future_df = pd.DataFrame(index=future_dates, data={'Predictions': future_predictions})

    # 3. Combine Historical and Future Data for plotting
    combined_df = pd.concat([df['Close'], future_df['Predictions']])

    plt.figure(figsize=(16, 8))
    plt.title(f'{STOCK_TICKER} Historical Data and {future_days}-Day Future Forecast', fontsize=20)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price (USD)', fontsize=18)

    # Plot historical data (Training data)
    plt.plot(df.index, df['Close'], color='blue', label='Historical Price')

    # Plot future forecast (Starting from the last known date)
    plt.plot(future_df.index, future_df['Predictions'], color='red', label='Future Forecast (LSTM)')

    plt.legend(loc='upper left', fontsize=14)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    try:
        # 1. Load Data
        df = load_data(STOCK_TICKER, START_DATE, END_DATE)

        # 2. Preprocess Data: Train on ALL data and get the last sequence
        X_all, y_all, scaler, full_df, last_known_sequence = preprocess_data(
            df, SEQUENCE_LENGTH
        )

        # 3. Build Model
        # input_shape is (time_steps, features) -> (60, 1)
        model = build_lstm_model((X_all.shape[1], 1))

        # 4. Train Model on ALL data for maximum context
        print("\n--- Starting Training on Full Historical Data ---")
        model.fit(X_all, y_all, batch_size=32, epochs=20, verbose=1)
        print("--- Training Complete ---")

        # 5. Make Future Predictions
        future_predictions = predict_future_prices(
            model,
            last_known_sequence,
            scaler,
            FUTURE_DAYS
        )

        # 6. Plotting - Plot historical data + the new future forecast
        plot_results(full_df, future_predictions, SEQUENCE_LENGTH, FUTURE_DAYS)

    except Exception as e:
        print(f"An error occurred: {e}")