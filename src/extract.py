import csv
import os
import tweepy
import pandas as pd
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
    alltweets = []
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1

    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name=screen_name, count=1000, max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1

    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]

    # with open('../output/%s_tweets.csv' % screen_name, 'w') as f:
    #   writer = csv.writer(f)
    #  writer.writerow(["id", "created_at", "text"])
    #  writer.writerows(outtweets)

    data_user = pd.DataFrame({'id': [tweet.id_str for tweet in alltweets],
                              'tweets': [tweet.text for tweet in alltweets],
                              'date_created': [tweet.created_at for tweet in alltweets],
                              })
    data_user.to_csv('../output/%s_tweets.csv' % screen_name)
