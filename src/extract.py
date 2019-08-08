import os

import pandas as pd
import tweepy
from dotenv import load_dotenv

load_dotenv()

consumer_key = os.environ["consumer_key"]
consumer_secret = os.environ["consumer_secret"]
access_token = os.environ["access_token"]
access_token_secret = os.environ["access_token_secret"]

# Authorization to use tweepy and extract information from a Twitter's user

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


def get_tweets_csv(screen_name):  # Function to extract all tweets and create a csv with them
    print("we are extracting the tweets...")
    alltweets = []
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1

    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name=screen_name, count=1000, max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1

    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]

    data_user = pd.DataFrame({'id': [tweet.id_str for tweet in alltweets],
                              'tweets': [tweet.text for tweet in alltweets],
                              'date_created': [tweet.created_at for tweet in alltweets],
                              })
    data_user.to_csv('../output/%s_tweets.csv' % screen_name)
