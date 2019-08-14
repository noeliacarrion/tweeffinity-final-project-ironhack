import argparse
import warnings

import get_similarity
from clustering import *
from transform import *


def get_args():
    parser = argparse.ArgumentParser(description="Username in Twitter")
    parser.add_argument('-u', '--user', help="type your username in Twitter", type=str)
    parser.add_argument('-f', '--friend', help="type your friend's username in Twitter", type=str)

    return parser.parse_args()


def main():
    warnings.filterwarnings("ignore")
    args = get_args()
    user1 = args.user
    user2 = args.friend
    # user1 = 'Noelia_Carrion9'
    # user2 = 'AdrianaRecio_'

    get_tweets_csv(user1)
    get_tweets_csv(user2)

    # From transform.py

    data_user = openData('../input/%s_tweets.csv' % user1)
    data_friend = openData('../input/%s_tweets.csv' % user2)

    data_user['RT'] = data_user['tweets'].apply(rtweet)
    data_user['mention'] = mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_user['hashtag'] = hashtag(data_user, 'tweets', 'r(\#\w+)')
    data_user['tokenized'] = data_user['tweets'].apply(spacyTokenizer)

    data_friend['RT'] = data_user['tweets'].apply(rtweet)
    data_friend['mention'] = mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_friend['hashtag'] = mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_friend['tokenized'] = data_friend['tweets'].apply(spacyTokenizer)

    data_user = dropRows(data_user, 'tokenized')
    data_friend = dropRows(data_friend, 'tokenized')

    data_user['date'] = createDate(data_user, 'date_created')
    data_friend['date'] = createDate(data_friend, 'date_created')

    data_user = dropcolumns(data_user, 'date_created')
    data_friend = dropcolumns(data_friend, 'date_created')

    wordCloud(user1, data_user['tokenized'])
    wordCloud(user2, data_friend['tokenized'])

    data_user = modifiedDate(data_user, 'date', 2016, 1, 1)
    data_friend = modifiedDate(data_friend, 'date', 2016, 1, 1)

    # From clustering.py

    tfidf_matrix_user = get_similarity.tfidfVectorizer(spacyTokenizer, 3, data_user['tweets'].values)
    tfidf_matrix_friend = get_similarity.tfidfVectorizer(spacyTokenizer, 3, data_friend['tweets'].values)

    data_embedding_user, embedding_user = get_similarity.embeddingUmap(2, 4, 42, tfidf_matrix_user, tfidf_matrix_user)
    data_embedding_friend, embedding_friend = get_similarity.embeddingUmap(2, 3, 42, tfidf_matrix_user,
                                                                           tfidf_matrix_friend)

    # plotEmbedding(data_embedding_user, data_embedding_friend, 'emb_1', 'emb_2', 'Spectral', 8, 'Umap Proyection', 12)

    cluster_user, label_user = get_similarity.clustering(embedding_user, embedding_user, 7, True)
    cluster_friend, label_friend = get_similarity.clustering(embedding_user, embedding_friend, 13, True)

    # plotClusterUser(data_embedding_user, 'emb_1', 'emb_2', cluster_user, label_user,
    # 'RdYlBu', 15, 12, "Tweets'cluster from user")

    # plotClusterFriend(data_embedding_friend, 'emb_1', 'emb_2', cluster_friend, label_friend,
    #     'RdYlBu', 15, 12, "Tweets'cluster from friend")

    # From get_similarity.py

    data_friend['cluster'] = get_similarity.addColum(data_friend, 'cluster', label_friend)
    data_friend = get_similarity.mergeDataframe(data_friend, data_embedding_friend)

    data_user['cluster'] = get_similarity.addColum(data_user, 'cluster', label_user)
    data_user = get_similarity.mergeDataframe(data_user, data_embedding_user)

    print(get_similarity.compareSimilarity(data_user, data_friend, 'emb_1', 'emb_2', 'cluster'))

    plotClusterTogether(data_embedding_user, data_embedding_friend, 'emb_1', 'emb_2', label_user, "k", 15,
                        12, "Tweets'cluster from user and friend")


if __name__ == '__main__':
    main()
