import asyncio
import time
from twikit import Client
from collections import Counter
from datetime import datetime, timedelta
import pandas as pd
import time

async def fetch_tweets(client, company_name, start_date, end_date, max_tweets=100):
    all_tweets = []
    
    # Initial search for tweets (set count to 20 as twikit restricts max batch size)
    tweets = await client.search_tweet(f'from:{company_name} since:{start_date} until:{end_date}', 'Latest', count=20)
    
    # Add the first batch of tweets to the list
    all_tweets.extend(tweet.created_at_datetime for tweet in tweets)
    
    # Continue fetching more tweets using pagination until you reach the max_tweets limit
    while len(all_tweets) < max_tweets:
        try:
            # Check if there are more tweets to fetch
            more_tweets = await tweets.next()
            if not more_tweets:
                break
            
            # Append tweets to the list
            all_tweets.extend(tweet.created_at_datetime for tweet in more_tweets)
            
            # Optional: sleep to avoid hitting rate limits
            time.sleep(1)
        
        except Exception as e:
            print(f"Error during pagination: {e}")
            break
    
    # Return the collected tweets
    return all_tweets


def get_tweet_count_per_day(tweets):
    # Extract only the date (YYYY-MM-DD) from each tweet's 'created_at' timestamp
    tweet_dates = [tweet.strftime('%Y-%m-%d') for tweet in tweets]
    
    # Use Counter to count the occurrences of each date
    tweet_count_by_day = Counter(tweet_dates)
    
    # Convert the counter to a DataFrame
    df = pd.DataFrame(tweet_count_by_day.items(), columns=['date', 'tweet_count'])
    
    # Sort the DataFrame by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date').reset_index(drop=True)
    
    return df


# Example usage
async def main():
    # Assuming the client is already logged in
    company_name = 'Tesla'
    start_date = '2024-09-01'
    end_date = '2024-09-30'
    client = Client('en-US')

    client.load_cookies('cookies.json')
    
    all_tweets = await fetch_tweets(client, company_name, start_date, end_date, max_tweets=200)

    # Get a DataFrame with the count of tweets per day
    df = get_tweet_count_per_day(all_tweets)
    
    # # Print the collected tweets and the count
    # for tweet in all_tweets:
    #     print(tweet)
    # print(f'Total tweets collected: {len(all_tweets)}')

    # Print the DataFrame
    print(df)

# Run the async function
asyncio.run(main())
