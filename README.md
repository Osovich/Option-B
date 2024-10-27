Before running the program, please run requirements.txt, it will install all the required libraries.

  pip install -r .\requirements.txt

Virtual Environments were ommitted for the project due to their slow performance compared to native python runtime speeds.

There are several parameters you can modify, these can be found at the start of stockprediction.py:
![image](https://github.com/user-attachments/assets/efc4da18-29c8-40bb-9096-6e3840535d66)

Company: string of the company name to predict for.<br />
Data_Start_Date and Data_End_Date: Timeframe for the training and testing. Format for it is YYYY-MM-DD.<br />
Save_data: Whether or not to save the dataset from yahoo finance.<br />
Prediction_days: how many days into the past to use to predict the future.<br />
Split_method: method to split the train test data, either Random or Date.<br />
Feature_columns: Columns for training and testing the model, highly important.<br />
Scale_features: Whether to scale features or not.<br />
Scale_min: minimum for scaling the data.<br />
Scale_max: maximum for scaling the data.<br />
Save_scalers: whether to save scalers for later.<br />
Prediction column: Specific column to predict in the model prediction.<br />
ML: Machine Learning model to use.<br />
Num_Layers: Number of layers to use.<br />
Layer_size: Size per layer to use.<br />
Tweets: Whether or not to use Tweets for prediction.<br />
Max_tweets: Maximum number of tweets allowed, keep under 500 to not break the Rate Limit.<br />

In order to use Tweets, you must use a Twitter account, and load the Cookies into a json file in the same folder as the program called cookies.json. The json file will have 2 keys, "auth_token" and ct0. To find these values, you must be logged into a Twitter account, press F12 and look for the Application Tab, then in the left list, enter Cookies, then X.com, and the first 2 values are the ones to use.<br />
![image](https://github.com/user-attachments/assets/c3eb1f32-e52f-4472-bc8f-bd29af32e532)

[Report 1.pdf](https://github.com/user-attachments/files/16637333/FrancoJimenez_104173896_B01.pdf)
