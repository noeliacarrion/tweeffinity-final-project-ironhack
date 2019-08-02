import re
import nltk
import pandas as pd

nltk.download('stopwords')


def open_data(file):
    df = pd.read_csv(file)
    return df


tweets_user = open_data('./final_project_ironhack/output/User_Noelia/Noelia_Carrion9_tweets.csv')


def rt_column(input_text): """create a column with value 1 to retweets and 0 to not retweets"""
    if input_text.startswith("RT"):
        value = 1
    else:
        value = 0
    return value


tweets_user['RT'] = tweets_user.text.apply(rt_column)

tweets_user['mention'] = tweets_user.text.str.findall(r'(?<![@\w])@(\w{1,25})').apply(', '.join)
tweets_user['hashtag'] = tweets_user.text.str.extract(r'(\#\w+)', expand=False)
tweets_user['hashtag'] = tweets_user['hashtag'].fillna("")


def remove_characters(input_text):
    list_remove = ["RT", "(?<![@\w])@(\w{1,25})", "\#+\w+"]
    for charac in list_remove:
        input_text = re.sub(charac, '', input_text)
    return input_text.lower()


def remove_special_characters(data, column):
    input_text = data[column].str.replace("[\d\W_]+", " ", re.UNICODE)
    return input_text


def deEmojify(inputString):
    return inputString.encode('latin-1', 'ignore').decode('latin-1')


tweets_user['cleaned_tweets'] = tweets_user.text.apply(remove_characters)
tweets_user['cleaned_tweets'] = tweets_user['cleaned_tweets'].str.replace('http\S+|www.\S+', '', case=False)
tweets_user['cleaned_tweets'] = remove_special_characters(tweets_user, "cleaned_tweets")
tweets_user['cleaned_tweets'] = tweets_user['cleaned_tweets'].apply(deEmojify)

stopwords_set = set(stopwords.words("spanish"))
cleaned_tweets = []
for index, row in tweets_user.iterrows():
    words_without_stopwords = [word for word in row.cleaned_tweets.split() if not word in stopwords_set]
    cleaned_tweets.append(' '.join(words_without_stopwords))
tweets_user['cleaned_tweets'] = cleaned_tweets

tweets_user['date'] = tweets_user['created_at']
tweets_user['date'] = pd.DatetimeIndex(tweets_user['created_at']).to_period('D')


def dropcolumns(data, columns):
    data = data.drop(columns, axis=1)
    return data

tweets_user = dropcolumns(tweets_user, ["created_at", "text"])

#tweets_user = tweets_user.drop(columns=["created_at", "text"])
tweets_user = tweets_user[["id", "date", "mention", "hashtag", "RT", "cleaned_tweets"]]

