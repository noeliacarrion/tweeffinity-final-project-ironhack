from transform import *


def main():
    user1 = 'Noelia_Carrion9'
    user2 = 'AdrianaRecio_'
    ''' 
    get_tweets_csv(user1)
    get_tweets_csv(user2) '''

    data_user = openData('../output/%s_tweets.csv' % user1)
    data_friend = openData('../output/%s_tweets.csv' % user2)

    # data_friend = open_data('../output/%s_tweets.csv' % user2)
    data_user['RT'] = data_user['tweets'].apply(rtweet)
    data_user['mention'] = mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_user['hashtag'] = hashtag(data_user, 'tweets', 'r(\#\w+)')
    data_user['tokenized'] = data_user['tweets'].apply(spacyTokenizer)
    data_user = dropRows(data_user, 'tokenized')

    data_friend['RT'] = data_user['tweets'].apply(rtweet)
    data_friend['mention'] = mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_friend['hashtag'] = mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_friend['tokenized'] = data_friend['tweets'].apply(spacyTokenizer)
    data_friend = dropRows(data_friend, 'tokenized')

    wordCloud(user1, data_user['tokenized'])
    wordCloud(user2, data_friend['tokenized'])

    print(data_friend['tokenized'])


if __name__ == '__main__':
    main()
