# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Description:
# This script loads stock data, preprocesses it, trains a Long Short-Term Memory (LSTM) model,
# and predicts future stock prices. It also includes functions for visualizing data using candlestick
# and boxplot charts. The code is designed to be modular and easily modifiable.

# Requirements:
# pip install numpy matplotlib pandas tensorflow scikit-learn yfinance mplfinance

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pickle
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
from sklearn.model_selection import train_test_split


def load_and_process_data(
    company, 
    start_date, 
    end_date,
    save_data,
    prediction_column, 
    prediction_days, 
    feature_columns=[], 
    split_method='date', 
    split_ratio=0.8, 
    split_date=None,
    scale_features=False, 
    scale_min=0, 
    scale_max=1, 
    save_scalers=False,
    data_dir='data'
): 
    """
    Load and process stock data with multiple features.

    Parameters:
        company (str): Company name.
        start_date (str): Start date for the dataset in 'YYYY-MM-DD' format.
        end_date (str): End date for the dataset in 'YYYY-MM-DD' format.
        save_data (bool): Whether to save the dataset to a file.
        prediction_column (str): Column name used for prediction.
        prediction_days (int): Number of days to predict into the future.
        feature_columns (list): List of feature columns to use in the model.
        split_method (str): Method to split the data into train/test sets ('date' or 'random').
        split_ratio (float): Ratio of train/test data if split_method is 'random'.
        split_date (str): Date to split the data if split_method is 'date'.
        scale_features (bool): Whether to scale the feature columns.
        scale_min (float): Minimum value to scale the feature columns.
        scale_max (float): Maximum value to scale the feature columns.
        save_scalers (bool): Whether to save the scalers to a file.
        data_dir (str): Directory to save the data.

    Returns:
        processed_data: Dictionary containing processed data and other relevant information.
    """

    # Create data directory if it doesn't exist
    if save_data and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # File path for saving/loading data
    file_path = os.path.join(data_dir, f'{company}_{start_date}_{end_date}.csv')
    
    # Load data
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    else:
        print(f"Fetching data for {company} from Yahoo Finance")
        data = yf.download(company, start=start_date, end=end_date)
        if save_data:
            data.to_csv(file_path)
    
    # Handle NaN values by dropping them
    data = data.dropna()
   
    # Store the original data and relevant details in a dictionary
    processed_data = {
        'df': data.copy(),
        'feature_columns': feature_columns if feature_columns else list(data.columns)
    }
    
    # Ensure all feature columns exist in the dataframe
    for col in feature_columns:
        assert col in data.columns, f"'{col}' does not exist in the dataframe."
    
    # Data splitting based on the method chosen
    if split_method == 'date':
        train_data = data.loc[data.index < split_date]
        test_data = data.loc[data.index >= split_date]
    elif split_method == 'random':
        train_data, test_data = train_test_split(data, train_size=split_ratio, random_state=42)
    
    # Reset index and sort data by date
    train_data = train_data.reset_index().sort_values(by='Date')
    test_data = test_data.reset_index().sort_values(by='Date')

    # Scale features if required
    if scale_features:
        scaler_dict = {}
        scaled_train_data = {}
        scaled_test_data = {}
        
        for col in feature_columns:
            scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
            scaled_train_data[col] = scaler.fit_transform(train_data[col].values.reshape(-1, 1)).ravel()
            scaled_test_data[col] = scaler.transform(test_data[col].values.reshape(-1, 1)).ravel()
            scaler_dict[col] = scaler

        if save_scalers:
            scalers_dir = os.path.join(os.getcwd(), 'scalers')
            if not os.path.exists(scalers_dir):
                os.makedirs(scalers_dir)
            scaler_file_path = os.path.join(scalers_dir, f"{company}_{start_date}_{end_date}_scalers.pkl")
            with open(scaler_file_path, 'wb') as f:
                pickle.dump(scaler_dict, f)
       
        train_data = pd.DataFrame(scaled_train_data)
        test_data = pd.DataFrame(scaled_test_data)
        
        processed_data["column_scaler"] = scaler_dict

    processed_data["scaled_train"] = train_data
    processed_data["scaled_test"] = test_data

    # Prepare training and testing datasets for LSTM
    x_train, y_train = [], []
    for i in range(prediction_days, len(train_data)):
        x_train.append(train_data[feature_columns].iloc[i-prediction_days:i])
        y_train.append(train_data[prediction_column].iloc[i])

    processed_data["x_train"] = np.array(x_train).reshape(-1, prediction_days, len(feature_columns))
    processed_data["y_train"] = np.array(y_train)
    
    x_test, y_test = [], []
    for i in range(prediction_days, len(test_data)):
        x_test.append(test_data[feature_columns].iloc[i-prediction_days:i])
        y_test.append(test_data[prediction_column].iloc[i])

    processed_data["x_test"] = np.array(x_test).reshape(-1, prediction_days, len(feature_columns))
    processed_data["y_test"] = np.array(y_test)

    return processed_data


def plot_candlestick_chart(df, title, n=1):
    """
    Plot a candlestick chart for the stock data.

    Parameters:
        df (pd.DataFrame): The stock data.
        title (str): Title of the chart.
        n (int): Number of trading days each candlestick should represent (default is 1).
    """
    if n > 1:
        # Resample the data to represent n trading days per candlestick
        df_resampled = df.resample(f'{n}D').agg({'Open': 'first',
                                                 'High': 'max',
                                                 'Low': 'min',
                                                 'Close': 'last',
                                                 'Volume': 'sum'})
    else:
        df_resampled = df
    
    mpf.plot(df_resampled, type='candle', volume=True, title=title)


def plot_boxplot_chart(df, title, n=10, label_interval=90):
    """
    Plot a boxplot chart for the stock data over a moving window.

    Parameters:
        df (pd.DataFrame): The stock data.
        title (str): Title of the chart.
        n (int): Size of the moving window in trading days (default is 10).
        label_interval (int): Interval between labels on the x-axis.
    """
    # Create the moving window data for the boxplots
    moving_windows = [df['Close'].iloc[i:i+n].tolist() for i in range(len(df) - n + 1)]

    # Generate dates for the x-axis labels
    dates = df.index[n-1:]

    # Plot the boxplot chart
    plt.figure(figsize=(12, 9))
    plt.boxplot(moving_windows, patch_artist=True)

    # Set title and labels
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')

    # Apply interval to x-axis labels
    plt.xticks(ticks=range(0, len(dates), label_interval), labels=dates.strftime('%Y-%m')[::label_interval], rotation=45)

    plt.show()


def create_dl_model(layer_type, num_layers, layer_size, input_shape, dropout_rate=0.2, optimizer = "adam", loss = "mean_squared_error"):
    """
    Function to dynamically create a deep learning model.

    Parameters:
        layer_type (str): The type of layer to use ('LSTM', 'GRU', 'RNN').
        num_layers (int): The number of layers in the model.
        layer_size (int): The number of units in each layer.
        input_shape (tuple): The shape of the input data (timesteps, features).
        dropout_rate (float): The dropout rate for regularization.
    
    Returns:
        model: Compiled deep learning model.
    """
    model = Sequential()
    
    # Dynamically add layers based on layer_type and number of layers
    for i in range(num_layers):
        if i == 0:
            # For First Layer
            model.add(layer_type(units=layer_size, return_sequences=True, batch_input_shape=input_shape))
        elif i == num_layers - 1:
            # For Last Layer
            model.add(layer_type(units=layer_size, return_sequences=False))
        else:
            # Black Box Layers
            model.add(layer_type(units=layer_size, return_sequences=True))

        # Add dropout for regularization
        model.add(Dropout(dropout_rate))
    
    # Final dense layer
    model.add(Dense(units=1, activation= "linear"))
    
    # Compile the model
    model.compile(loss=loss, metrics = ["mean_squared_error"], optimizer=optimizer)
    
    return model


# Define parameters
COMPANY = "CBA.AX"  
DATA_START_DATE = '2020-01-01'
DATA_END_DATE = '2022-12-31'
SAVE_DATA = True
PREDICTION_DAYS = 20
SPLIT_METHOD = 'random'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2021-01-02'
# FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
FEATURE_COLUMNS = ['Close']
SCALE_FEATURES = True
SCALE_MIN = 0
SCALE_MAX = 1
SAVE_SCALERS = True
PREDICTION_COLUMN = "Close"

# Load and process data
data = load_and_process_data(
    company=COMPANY, 
    start_date=DATA_START_DATE, 
    end_date=DATA_END_DATE, 
    save_data=SAVE_DATA,
    prediction_column=PREDICTION_COLUMN,
    prediction_days=PREDICTION_DAYS,
    split_method=SPLIT_METHOD, 
    split_ratio=SPLIT_RATIO, 
    split_date=SPLIT_DATE,
    feature_columns=FEATURE_COLUMNS,
    scale_features=SCALE_FEATURES,
    scale_min=SCALE_MIN,
    scale_max=SCALE_MAX,
    save_scalers=SAVE_SCALERS
)

# Plot candlestick chart
plot_candlestick_chart(data['df'], title=f"{COMPANY} Candlestick Chart", n=5)

# Plot boxplot chart
plot_boxplot_chart(data['df'], title=f"{COMPANY} Boxplot Chart", n=10)

layer_type = SimpleRNN  # Can be 'LSTM', 'GRU', or 'SimpleRNN'
num_layers = 3
layer_size = 50
input_shape = (None, data["x_train"].shape[1], len(FEATURE_COLUMNS))  # Timesteps and features


# Model creation
# model = Sequential()

# # LSTM layers with dropout for regularization
# model.add(LSTM(units=50, return_sequences=True, input_shape=(data["x_train"].shape[1], 1)))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=50))
# model.add(Dropout(0.2))

# # Dense layer for final prediction output
# model.add(Dense(units=1))

# # Compile model with optimizer and loss function
# model.compile(optimizer='adam', loss='mean_squared_error')

# Create and compile the model
model = create_dl_model(layer_type, num_layers, layer_size, input_shape)

# Train the model
model.fit(data["x_train"], data["y_train"], epochs=10, batch_size=32)

# Test the model and plot predictions
actual_prices = data["column_scaler"][PREDICTION_COLUMN].inverse_transform(data["y_test"].reshape(-1, 1))
predicted_prices = model.predict(data['x_test'])
predicted_prices = data["column_scaler"][PREDICTION_COLUMN].inverse_transform(predicted_prices)

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Save predictions
pd.DataFrame(predicted_prices, columns=["Predicted Prices"]).to_csv('predicted_prices.csv', index=False)
pd.DataFrame(actual_prices, columns=["Actual Prices"]).to_csv('actual_prices.csv', index=False)

# Predict the next day's stock price
real_data = [data["x_test"][len(data['x_test']) - 1]]
real_data = np.array(real_data)

# Make prediction for the next day
predicted_next_day_price = model.predict(real_data)
predicted_next_day_price = data["column_scaler"][PREDICTION_COLUMN].inverse_transform(predicted_next_day_price)

print(f"Predicted next day price: {predicted_next_day_price[0][0]}")
