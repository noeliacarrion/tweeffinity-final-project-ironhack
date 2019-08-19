import argparse
import warnings
import get_similarity
import transform


def get_args():  # add two arguments to run the program: user1 and user2
    parser = argparse.ArgumentParser(description="Username in Twitter")
    parser.add_argument('-u', '--user', help="type your username in Twitter", type=str)
    parser.add_argument('-f', '--friend', help="type your friend's username in Twitter", type=str)

    return parser.parse_args()


def functions():
    warnings.filterwarnings("ignore")
    args = get_args()
    user1 = args.user
    user2 = args.friend

    get_similarity.get_tweets_csv(user1)
    get_similarity.get_tweets_csv(user2)

    # From transform.py

    data_user = transform.openData('../output/%s_tweets.csv' % user1)
    data_friend = transform.openData('../output/%s_tweets.csv' % user2)

    data_user['RT'] = data_user['tweets'].apply(transform.rtweet)
    data_user['mention'] = transform.mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_user['hashtag'] = transform.hashtag(data_user, 'tweets', 'r(\#\w+)')
    data_user['tokenized'] = data_user['tweets'].apply(transform.spacyTokenizer)

    data_friend['RT'] = data_user['tweets'].apply(transform.rtweet)
    data_friend['mention'] = transform.mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_friend['hashtag'] = transform.mention(data_user, 'tweets', 'r(?<![@\w])@(\w{1,25})')
    data_friend['tokenized'] = data_friend['tweets'].apply(transform.spacyTokenizer)

    data_user = transform.dropRows(data_user, 'tokenized')
    data_friend = transform.dropRows(data_friend, 'tokenized')

    data_user['date'] = transform.createDate(data_user, 'date_created')
    data_friend['date'] = transform.createDate(data_friend, 'date_created')

    data_user = transform.dropcolumns(data_user, 'date_created')
    data_friend = transform.dropcolumns(data_friend, 'date_created')

    transform.wordCloud(user1, data_user['tokenized'])
    transform.wordCloud(user2, data_friend['tokenized'])

    data_user = transform.modifiedDate(data_user, 'date', 2016, 1, 1)
    data_friend = transform.modifiedDate(data_friend, 'date', 2016, 1, 1)

    # From clustering.py

    tfidf_matrix_user = get_similarity.tfidfVectorizer(transform.spacyTokenizer, 3, data_user['tweets'].values)
    tfidf_matrix_friend = get_similarity.tfidfVectorizer(transform.spacyTokenizer, 3, data_friend['tweets'].values)

    data_embedding_user, embedding_user = get_similarity.embeddingUmap(2, 4, 42, tfidf_matrix_user, tfidf_matrix_user)
    data_embedding_friend, embedding_friend = get_similarity.embeddingUmap(2, 3, 42, tfidf_matrix_user,
                                                                           tfidf_matrix_friend)

    cluster_user, label_user = get_similarity.clustering(embedding_user, embedding_user, 7, True)
    cluster_friend, label_friend = get_similarity.clustering(embedding_user, embedding_friend, 13, True)

    # From get_similarity.py

    data_friend['cluster'] = get_similarity.addColum(data_friend, 'cluster', label_friend)
    data_friend = get_similarity.mergeDataframe(data_friend, data_embedding_friend)

    data_user['cluster'] = get_similarity.addColum(data_user, 'cluster', label_user)
    data_user = get_similarity.mergeDataframe(data_user, data_embedding_user)

    print(get_similarity.compareSimilarity(data_user, data_friend, 'emb_1', 'emb_2', 'cluster'))

    get_similarity.plotClusterTogether(data_embedding_user, data_embedding_friend, 'emb_1', 'emb_2', label_user, "k", 15,
                                       12, "Tweets'cluster from user and friend")
