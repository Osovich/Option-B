Before running the program, please run requirements.txt, it will install all the required libraries.

  pip install -r .\requirements.txt

Virtual Environments were ommitted for the project due to their slow performance compared to native python runtime speeds.

There are several parameters you can modify, these can be found at the start of stockprediction.py:
![image](https://github.com/user-attachments/assets/efc4da18-29c8-40bb-9096-6e3840535d66)

Company: string of the company name to predict for.
Data_Start_Date and Data_End_Date: Timeframe for the training and testing. Format for it is YYYY-MM-DD
Save_data: Whether or not to save the dataset from yahoo finance.
Prediction_days: how many days into the past to use to predict the future.
Split_method: method to split the train test data, either Random or Date.
Feature_columns: Columns for training and testing the model, highly important.
Scale_features: Whether to scale features or not
Scale_min: minimum for scaling the data.
Scale_max: maximum for scaling the data.
Save_scalers: whether to save scalers for later.
Prediction column: Specific column to predict in the model prediction.
ML: Machine Learning model to use.
Num_Layers: Number of layers to use.
Layer_size: Size per layer to use.
Tweets: Whether or not to use Tweets for prediction.
Max_tweets: Maximum number of tweets allowed, keep under 500 to not break the Rate Limit.
[Report 1.pdf](https://github.com/user-attachments/files/16637333/FrancoJimenez_104173896_B01.pdf)
