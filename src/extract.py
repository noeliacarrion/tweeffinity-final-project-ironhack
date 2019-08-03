import csv
import os
import time
import tweepy
from dotenv import load_dotenv
load_dotenv()

consumer_key = os.environ["consumer_key"]
consumer_secret = os.environ["consumer_secret"]
access_token = os.environ["access_token"]
access_token_secret = os.environ["access_token_secret"]

#"""Authorization to use Tweepy and extract information from a Twitter's user"""
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)


def get_tweets_csv(screen_name): #"""Function to extract all tweets and create a csv with them"""
    alltweets = []
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)
    alltweets.extend(new_tweets)
    oldest = alltweets[-1].id - 1

    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name=screen_name, count=1000, max_id=oldest)
        alltweets.extend(new_tweets)
        oldest = alltweets[-1].id - 1

    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]

    with open('../output/User_Noelia/%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "created_at", "text"])
        writer.writerows(outtweets)

def get_followers(user_id): #"""function to extract all followers from the user"""
    list_users = []
    while True:
        try:
            for i, user in enumerate(tweepy.Cursor(api.followers, id=user_id, count=200).pages()):
                list_users += user
            user_final = [[user.id_str, user.created_at, user.name, user.screen_name, user.location, user.description,
                           user.followers_count, user.friends_count] for user in list_users]
            with open("../output/User_Noelia/%s_followers.csv" % user_id, 'w') as f:
                wr = csv.writer(f)
                wr.writerows([["id", "created_at", "name", "screen_name", "location", "description", "followers_count",
                               "friends_count"]])
                wr.writerows(user_final)
            pass
        except tweepy.TweepError:
            time.sleep(60 * 15)
            continue
        break


def get_friends(user_id): #"""function to extract all friends(people to whom the user follows) and create a csv with
                            #the information"""
    list_users = []
    while True:
        try:
            for i, user in enumerate(tweepy.Cursor(api.friends, id=user_id, count=200).pages()):
                list_users += user
            user_final = [[user.id_str, user.created_at, user.name, user.screen_name, user.location, user.description,
                           user.followers_count, user.friends_count] for user in list_users]
            with open("../output/User_Noelia/%s_friends.csv" % user_id, 'w') as f:
                wr = csv.writer(f)
                wr.writerows([["id", "created_at", "name", "screen_name", "location", "description", "followers_count",
                               "friends_count"]])
                wr.writerows(user_final)
            pass
        except tweepy.TweepError:
            time.sleep(60 * 15)
            continue
        break

get_tweets_csv('@mrogati')
# get_friends('Noelia_Carrion9')
# get_followers('Noelia_Carrion9')
