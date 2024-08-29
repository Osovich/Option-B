# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf
import pickle



from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from sklearn.model_selection import train_test_split


#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo

def load_and_process_data(
    ticker, 
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
    data_dir='data'): 
    """
    Load and process stock data with multiple features.
    
    Parameters:
    - ticker company ticker symbol
    - start_date start date for the dataset in the format 'YYYY-MM-DD'
    - end_date end date for the dataset in the format 'YYYY-MM-DD'
    - save_data, whether to save the dataset to a file
    - prediction_column, 
    - prediction_days, number of days to predict into the future
    - feature_columns=[], list of feature columns to use in the model
    - split_method='date' method to split the data into train/test data ('date' or 'random')
    - split_ratio=0.8, ratio of train/test data if split_method is 'random'
    - split_date=None date to split the data if split_method is 'date'
    - fillna_method='drop' method to drop or fill NaN values in the data ('drop', 'ffill', 'bfill', or 'mean')
    - scale_features=False, whether to scale the feature columns
    - scale_min=0, minimum value to scale the feature columns
    - scale_max=1, maximum value to scale the feature columns
    - save_scalers=False whether to save the scalers to a file
    
    Returns:
    - object all = {}
    """
    
     # Create data directory if not exists
    if save_data and not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # File path for saving/loading data
    file_path = os.path.join(data_dir, f'{ticker}_{start_date}_{end_date}.csv')
    
    # Load data
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        # Read from CSV if the data exists.
        data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    else:
        print(f"Fetching data for {ticker} from Yahoo Finance")
        data = yf.download(ticker, start=start_date, end=end_date)
        if save_data:
            data.to_csv(file_path)
    
    # Handle NaN values
    data = data.dropna()
   
    # this will contain all the elements we want to return from this function
    all = {}
    # we will also return the original dataframe itself
    all['df'] = data.copy()
   
    # make sure that the passed feature_columns exist in the dataframe
    if len(feature_columns) > 0:
        for col in feature_columns:
            assert col in data.columns, f"'{col}' does not exist in the dataframe."
    else:
        # if no feature_columns are passed, use all columns except the prediction_column
        feature_columns = list(filter(lambda column: column != 'Date', data.columns))
    
    # add feature columns to all
    all['feature_columns'] = feature_columns
    # Deal with potential NaN values in the data

    # Split data into train and test sets based on date
    if split_method == 'date':
        train = data.loc[data['Date'] < split_date]
        test = data.loc[data['Date'] >= split_date]
    # Split data into train and test sets randomly with provided ratio
    elif split_method == 'random':
        train, test = train_test_split(data, train_size=split_ratio, random_state=42)
    

    
    # Reset index of both dataframes
    train = train.reset_index()
    test = test.reset_index()
    # Sort dataframes by date
    train = train.sort_values(by='Date')
    test = test.sort_values(by='Date')

    # Scale features
    if scale_features:
        # Create scaler dictionary to store all scalers for each feature column
        scaler_dict = {}
        # Dictionaries to store scaled train and test data
        scaled_train = {}
        scaled_test = {}
        #loop through each feature column
        for col in feature_columns:
            # Create scaler for each feature column using Min Max, passing in the scale_min and scale_max
            scaler = MinMaxScaler(feature_range=(scale_min, scale_max))
            # Fit and transform scaler on train data
            scaled_train[col] = scaler.fit_transform(train[col].values.reshape(-1, 1)).ravel()
            # Transform test data using scaler
            scaled_test[col] = scaler.transform(test[col].values.reshape(-1,1)).ravel()
            # Add scaler to scaler dictionary, using the feature column name as key
            scaler_dict[col] = scaler
        # Add scaler dictionary to all
        all["column_scaler"] = scaler_dict
        
         # Save scalers to file
        if save_scalers:
            # Create scalers directory if it doesn't exist
            scalers_dir = os.path.join(os.getcwd(), 'scalers')
            if not os.path.exists(scalers_dir):
                os.makedirs(scalers_dir)
            # Create scaler file name
            scaler_file_name = f"{ticker}_{start_date}_{end_date}_scalers.txt"
            scaler_file_path = os.path.join(scalers_dir, scaler_file_name)
            with open(scaler_file_path, 'wb') as f:
                pickle.dump(scaler_dict, f)
       
        # Convert scaled data to dataframes
        train = pd.DataFrame(scaled_train)
        test = pd.DataFrame(scaled_test)

    # Add train and test data to all
    all["scaled_train"] = train
    all["scaled_test"] = test
    # Construct the X's and y's for the training data
    x_train, y_train = [], []
    # Loop through the training data from prediction_days to the end
    for x in range(prediction_days, len(train)):
        # Append the values of the passed prediction column to x_train and y_train
        x_train.append(train[prediction_column].iloc[x-prediction_days:x])
        y_train.append(train[prediction_column].iloc[x])
        
    # convert to numpy arrays
    all["x_train"] = np.array(x_train)
    all["y_train"] = np.array(y_train)
    # reshape x_train for proper fitting into LSTM model
    all["x_train"] = np.reshape(all["x_train"], (all["x_train"].shape[0], all['x_train'].shape[1], -1))
    # construct the X's and y's for the test data
    X_test, y_test = [], []
    # Loop through the test data from prediction_days to the end
    for x in range(prediction_days, len(test)):
        # Append the values of the passed prediction column to X_test and y_test
        X_test.append(test[prediction_column].iloc[x - prediction_days:x])
        y_test.append(test[prediction_column].iloc[x])

    # convert to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    #assign y_test to all
    all["y_test"] = y_test
    #assign X_test to all and reshape X_test for prediction compatibility
    all["X_test"] = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return all


# define function parameters to use
DATA_SOURCE = "yahoo"
COMPANY = "CBA.AX"  
DATA_START_DATE = '2015-01-01'
DATA_END_DATE = '2022-12-31'
SAVE_DATA = True
PREDICTION_DAYS = 100
SPLIT_METHOD = 'random'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2020-01-02'
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
SCALE_FEATURES = True
SCALE_MIN = 0
SCALE_MAX = 1
SAVE_SCALERS = True
prediction_column = "Close"

# Call processData function passing in parameters
data = load_and_process_data(
    ticker=COMPANY, 
    start_date=DATA_START_DATE, 
    end_date=DATA_END_DATE, 
    save_data=SAVE_DATA,
    prediction_column=prediction_column,
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




model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(data["x_train"].shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(data["x_train"], data["y_train"], epochs=25, batch_size=32)


#testing the model
actual_prices = data["column_scaler"][prediction_column].inverse_transform(data["y_test"].reshape(-1,1))
predicted_prices = model.predict(data['x_test'])

predicted_prices = data["column_scaler"][prediction_column].inverse_transform(predicted_prices)
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()


predicted_prices = predicted_prices.ravel()
actual_prices = actual_prices.ravel()
df = pd.DataFrame(predicted_prices)
df.to_csv('predicted_prices.csv', index=False)
df = pd.DataFrame(actual_prices)
df.to_csv('actual_prices.csv', index=False)
#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [data["x_test"][len(data['x_test']) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = data["column_scaler"][prediction_column].inverse_transform(prediction)
print(f"Prediction: {prediction[0]}")

