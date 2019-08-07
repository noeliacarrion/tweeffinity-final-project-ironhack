from transform import *
from clustering import *


def main():
    user1 = 'Noelia_Carrion9'
    user2 = 'AdrianaRecio_'
    '''
    get_tweets_csv(user1)
    get_tweets_csv(user2) 
'''
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

    tfidf_matrix_user = tfidfVectorizer(spacyTokenizer, 3, data_user['tweets'].values)
    tfidf_matrix_friend = tfidfVectorizer(spacyTokenizer, 3, data_friend['tweets'].values)

    data_embedding_user, embedding_user = embeddingUmap(2, 8, 42, tfidf_matrix_user, tfidf_matrix_user)
    data_embedding_friend, embedding_friend = embeddingUmap(2, 8, 42, tfidf_matrix_user, tfidf_matrix_friend)

    plotEmbedding(data_embedding_user, data_embedding_friend, 'emb_1', 'emb_2', 'Spectral', 8, 'Umap Proyection', 12)

    cluster_user, label_user = clustering(embedding_user, embedding_user, 15, True)
    cluster_friend, label_friend = clustering(embedding_user, embedding_friend, 15, True)

    plotClusterTogether(data_embedding_user, data_embedding_friend, 'emb_1', 'emb_2', "k", "g", 15,
                        12, "Tweets'cluster from user and friend")
    plotClusterUser(data_embedding_user, 'emb_1', 'emb_2', cluster_user, label_user,
                    'RdYlBu', 15, 12, "Tweets'cluster from user")

    plotClusterFriend(data_embedding_friend, 'emb_1', 'emb_2', cluster_friend, label_friend,
                      'RdYlBu', 15, 12, "Tweets'cluster from user")

    print(cluster_user, label_user, cluster_friend, label_friend)


''''

    wordCloud(user1, data_user['tokenized'])
    wordCloud(user2, data_friend['tokenized'])

    '''

if __name__ == '__main__':
    main()
