import csv
import os
import time
import tweepy
import networkx as nx
from dotenv import load_dotenv
load_dotenv()

consumer_key = os.environ["consumer_key"]
consumer_secret = os.environ["consumer_secret"]
access_token = os.environ["access_token"]
access_token_secret = os.environ["access_token_secret"]

#Authorization to use Tweepy and extract information from a Twitter's user
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)

graph = nx.DiGraph()
username="Noelia_Carrion9"

# Load data
print ("Loading users who follow", username, "...")


def get_friends(user_id):  # function to extract all friends(people to whom the user follows) and create a csv with
    list_users = []
    while True:
        try:
            for i, user in enumerate(tweepy.Cursor(api.friends, id=user_id, count=200).pages()):
                list_users += user
            user_final = [[user.id_str, user.created_at, user.name, user.screen_name, user.location, user.description,
                        user.followers_count, user.friends_count] for user in list_users]
            with open("../output/User_csv/%s_friends_prueba.csv" % user_id, 'w') as f:
                wr = csv.writer(f)
                wr.writerows([["id", "created_at", "name", "screen_name", "location", "description", "followers_count",
                           "friends_count"]])
                wr.writerows(user_final)
        except tweepy.TweepError:
            time.sleep(60 * 15)
            continue
    return list_users

print(get_friends('Noelia_Carrion9'))
