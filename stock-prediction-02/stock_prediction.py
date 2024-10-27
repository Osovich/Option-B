import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pickle
import mplfinance as mpf
from twikit import Client
from collections import Counter
import time
from datetime import timedelta

import asyncio



from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima  # For auto ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pmdarima.arima import ADFTest

from sklearn.metrics import mean_squared_error

# Define parameters
COMPANY = "TSLA"  
DATA_START_DATE = '2015-01-01'
DATA_END_DATE = '2022-12-31'
SAVE_DATA = True
PREDICTION_DAYS = 60
SPLIT_METHOD = 'random'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2021-06-01'
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
SCALE_FEATURES = True
SCALE_MIN = 0
SCALE_MAX = 1
SAVE_SCALERS = True
PREDICTION_COLUMN = "Close"
EPOCHS = 20
BATCH_SIZE = 64
PREDICTION_STEPS = 10
ML = SimpleRNN
NUM_LAYERS = 3
LAYER_SIZE = 50
TWEETS = True
MAX_TWEETS = 200

async def fetch_tweets(client, company_name, start_date, end_date, max_tweets=500):
    all_tweets = []
    
    # Convert the start and end dates to datetime objects
    current_start_date = pd.to_datetime(end_date)
    current_start_date -= timedelta(days=1)
    current_end_date = pd.to_datetime(end_date)
    global_start_date = pd.to_datetime(DATA_START_DATE)
    
    # Initial search for tweets
    while len(all_tweets) < max_tweets and current_start_date >= global_start_date:
        tweets = await client.search_tweet(f'from:${company_name} since:{current_start_date.strftime("%Y-%m-%d")} until:{current_end_date.strftime("%Y-%m-%d")}', 'Latest', count=20)
        print(f'from:${company_name} since:{current_start_date.strftime("%Y-%m-%d")} until:{current_end_date.strftime("%Y-%m-%d")}')
        # Add tweets to the list
        all_tweets.extend(tweet.created_at_datetime for tweet in tweets)
        
        # Decrement the start and end date by one day
        current_start_date -= timedelta(days=1)
        current_end_date -= timedelta(days=1)
        
        # sleep to avoid hitting rate limits
        time.sleep(1)
        
        # Break if no tweets are returned (can also use other break conditions)
        if not tweets:
            break
    
    return all_tweets


def get_tweet_count_per_day(tweets):
    # Extract only the date (YYYY-MM-DD) from each tweet's 'created_at' timestamp
    tweet_dates = [tweet.strftime('%Y-%m-%d') for tweet in tweets]
    
    # Use Counter to count the occurrences of each date
    tweet_count_by_day = Counter(tweet_dates)
    
    # Convert the counter to a DataFrame
    df = pd.DataFrame(tweet_count_by_day.items(), columns=['Date', 'Tweet_count'])
    
    # Sort the DataFrame by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    print(df)
    
    return df


async def load_and_process_data(
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
    data_dir='data',
    tweets=False
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
        tweets (bool): Wether or not to fetch tweets for training and testing.

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
    
    if tweets:
        FEATURE_COLUMNS.append('Tweet_count')
        # Load Tweets and make dataframe
        client = Client('en-US')

        client.load_cookies('cookies.json')
        
        all_tweets = await fetch_tweets(client, company, start_date, end_date, max_tweets=MAX_TWEETS)

        # Get a DataFrame with the count of tweets per day
        tweets_df = get_tweet_count_per_day(all_tweets)
        tweets_df['Date'] = pd.to_datetime(tweets_df['Date'])

        data['Date'] = pd.to_datetime(data.index)

        data = pd.merge(data.reset_index(drop=True), tweets_df, on='Date', how='left').set_index('Date')

    # Handle NaN values by filling them with 0
    data = data.fillna(0)
    print(data)
   
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

    # Prepare training and testing datasets for DL
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

def multi_step_predict(model, last_n_days, prediction_steps, feature_columns, scaler, prediction_column='Close'):
    """
    Predicts multiple future 'Close' prices (steps) based on the last n days of input data with multiple features.

    Parameters:
        model: Trained deep learning model for stock price prediction.
        last_n_days (np.array): The last n days of data (features) to use for prediction, shape (1, n, features).
        prediction_steps (int): Number of steps (days) to predict into the future.
        feature_columns (list): List of feature column names.
        scaler (sklearn scaler): Scaler used to inverse transform the predictions.
        prediction_column (str): The target column to predict ('Close').

    Returns:
        future_prices (list): List of predicted 'Close' prices for the next `prediction_steps` days.
    """
    current_input = last_n_days.copy()  # Shape: (1, n, features)
    future_prices = []

    for _ in range(prediction_steps):
        # Predict the next 'Close' price
        next_price_scaled = model.predict(current_input)  # Shape: (1, 1)

        # Inverse transform to get the actual price
        next_price = scaler.inverse_transform(next_price_scaled)[0][0]
        future_prices.append(next_price)

        # Prepare the next input sequence
        # Create a new input array where 'Close' is updated with the predicted price
        # and other features can be carried forward or set as needed.
        next_price_scaled_reshaped = next_price_scaled.reshape((1, 1, 1))  # Shape: (1, 1, 1)

        # If there are other features, decide how to handle them.
        # Here, we'll carry forward the last known values for other features.
        # Extract the last day's data
        last_day = current_input[0, -1, :].copy()  # Shape: (features,)

        # Update the 'Close' feature with the predicted price
        close_index = feature_columns.index(prediction_column)
        last_day[close_index] = next_price_scaled[0][0]

        # Reshape to (1, 1, features)
        new_day = last_day.reshape((1, 1, -1))

        # Append the new_day to current_input and remove the first day to maintain the sequence length
        current_input = np.concatenate((current_input[:, 1:, :], new_day), axis=1)

    return future_prices



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
        layer_type (import): The type of layer to use ('LSTM', 'GRU', 'SimpleRNN').
        num_layers (int): The number of layers in the model.
        layer_size (int): The number of units in each layer.
        input_shape (tuple): The shape of the input data (timesteps, features).
        dropout_rate (float): The dropout rate for regularization.
        optimizer (str): The optimization technique used, adam by default.
        loss (str): The loss technique used. From class, the one preferred by me was mean_squared_error.
        
    Returns:
        model: Compiled deep learning model.
    """
    model = Sequential()
    
    # Dynamically add layers based on layer_type and number of layers
    for i in range(num_layers):
        if i == 0:
            # For First Layer
            layer = layer_type(units=layer_size, return_sequences=True, batch_input_shape=input_shape)
        elif i == num_layers - 1:
            # For Last Layer
            layer = layer_type(units=layer_size, return_sequences=False)
        else:
            # Black Box Layers
            layer = layer_type(units=layer_size, return_sequences=True)

        model.add(layer)

        # Add dropout for regularization
        model.add(Dropout(dropout_rate))
    
    # Final dense layer
    model.add(Dense(units=1, activation= "linear"))
    
    # Compile the model
    model.compile(loss=loss, metrics = ["mean_squared_error"], optimizer=optimizer)
    
    return model
def find_sarima_params(data, seasonal_period):
    # Perform auto_arima to find the best order and seasonal_order
    model = auto_arima(
        data,
        seasonal=True,
        m=seasonal_period,  # Set seasonal period (e.g., 7 for weekly data)
        start_p=1, start_q=1, max_p=3, max_d=2,max_q=3,
        start_P=0,start_Q=0,max_P=3, max_D=3,max_Q=3,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        trace=True
    )

    # Return the best order and seasonal order
    return model.order, model.seasonal_order

async def main():
    # Load and process data
    data = await load_and_process_data(
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
        save_scalers=SAVE_SCALERS,
        tweets=TWEETS
    )

    # # Plot candlestick chart
    # plot_candlestick_chart(data['df'], title=f"{COMPANY} Candlestick Chart", n=5)

    # # Plot boxplot chart
    # plot_boxplot_chart(data['df'], title=f"{COMPANY} Boxplot Chart", n=10)

    layer_type = ML  # Can be 'LSTM', 'GRU', or 'SimpleRNN'
    num_layers = NUM_LAYERS
    layer_size = LAYER_SIZE
    input_shape = (1, data["x_train"].shape[1], len(FEATURE_COLUMNS))  # Timesteps and features


    # Create and compile the model
    model = create_dl_model(layer_type, num_layers, layer_size, input_shape)

    # Train the model
    model.fit(data["x_train"], data["y_train"], epochs=EPOCHS, batch_size=BATCH_SIZE)

    # Test the model and plot predictions
    actual_prices = data["column_scaler"][PREDICTION_COLUMN].inverse_transform(data["y_test"].reshape(-1, 1))
    predicted_prices = model.predict(data['x_test'])

    predicted_prices = data["column_scaler"][PREDICTION_COLUMN].inverse_transform(predicted_prices)



    # Save predictions
    pd.DataFrame(predicted_prices, columns=["Predicted Prices"]).to_csv('predicted_prices.csv', index=False)
    pd.DataFrame(actual_prices, columns=["Actual Prices"]).to_csv('actual_prices.csv', index=False)

    order, seasonal_order = find_sarima_params(data['df'][PREDICTION_COLUMN], seasonal_period=1)

    sarima_model = SARIMAX(data['df'][PREDICTION_COLUMN], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    sarima_fit = sarima_model.fit()
    sarima_fitted_values = sarima_fit.fittedvalues

    sarima_predictions = sarima_fit.forecast(steps=PREDICTION_STEPS)


    last_n_days = data["x_test"][-1].reshape((input_shape))
    future_prices = multi_step_predict(
        model, 
        last_n_days, 
        PREDICTION_STEPS, 
        feature_columns=FEATURE_COLUMNS, 
        scaler=data["column_scaler"][PREDICTION_COLUMN],
        prediction_column=PREDICTION_COLUMN
    )

    print(f"Predicted future 'Close' prices for the next {PREDICTION_STEPS} days:")
    for i, price in enumerate(future_prices, 1):
        print(f"Day {i}: {price:.2f}")

    ensemble_predictions = [(dl + sarima) / 2 for dl, sarima in zip(future_prices, sarima_predictions)]

    print(f"Predicted DL + SARIMA's next {PREDICTION_STEPS} days:")
    for i, price in enumerate(ensemble_predictions, 1):
        print(f"Day {i}: {price:.2f}")



    # Define future dates using the converted Timestamp
    future_dates = pd.date_range(start=data['df'].index[-1], periods=PREDICTION_STEPS + 1, freq='B')[1:]


    # Shape the sarima fitted values to be the length of the actual prices, and create a line for the ensemble of
    # it with the dl model training results
    sarima_fitted_values = sarima_fitted_values[-len(actual_prices):]
    ensemble_values = [(dl + sarima) / 2 for dl, sarima in zip(predicted_prices, sarima_fitted_values)]

    # Plot the original prices (actual and predicted on test data), then plot the sarima and ensemble training
    plt.figure(figsize=(12, 6))
    plt.plot(data['df'].index[-len(actual_prices):], actual_prices, color="black", label=f"Actual {COMPANY} Price")
    plt.plot(data['df'].index[-len(predicted_prices):], predicted_prices, color="green", label=f"Trained DL {COMPANY} Price")
    plt.plot(data['df'].index[-len(sarima_fitted_values):], sarima_fitted_values, color="pink", label=f"Trained SARIMA {COMPANY} Price")
    plt.plot(data['df'].index[-len(ensemble_values):], ensemble_values, color="brown", label=f"Trained SARIMA + DL {COMPANY} Price")


    # Add future DL predictions, SARIMA predictions, and ensemble predictions
    plt.plot(future_dates, future_prices, color="blue", label="Future DL Predictions", linestyle='dashed')
    plt.plot(future_dates, sarima_predictions, color="orange", label="SARIMA Predictions", linestyle='dashed')
    plt.plot(future_dates, ensemble_predictions, color="purple", label="Ensemble Predictions (DL + SARIMA)", linestyle='dashed')

    # Add plot title, axis labels, and legend
    plt.title(f"{COMPANY} Share Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(f"{COMPANY} Share Price")
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

asyncio.run(main())
