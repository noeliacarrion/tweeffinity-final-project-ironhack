import sklearn.feature_extraction.text
from umap import UMAP
import matplotlib.pyplot as plt
import numpy as np
from hdbscan import HDBSCAN
from extract import *


def tfidfVectorizer(spacy, min_df, values):
    tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=spacy, min_df=min_df).fit(values)
    print("creating tweet's vectors...")
    tfidf_matrix = tfidf_vectorizer.transform(values)

    return tfidf_matrix


def embeddingUmap(n_components, n_neighbors, random_state, tfidf_matrix_fit, tfidf_matrix_transform):
    umap = UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state).fit(tfidf_matrix_fit)
    print("reducing vector's dimensionality...")
    umap_embedding = umap.fit_transform(tfidf_matrix_transform)
    umap_df = pd.DataFrame(umap_embedding, columns=[f'emb_{i + 1}' for i in range(n_components)])

    return umap_df, umap_embedding


def plotEmbedding(embedding_user, embedding_friend, colum1, colum2, cmap, s, title, fontsize):
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding_user[colum1], embedding_user[colum2], cmap=cmap, s=s)
    plt.scatter(embedding_friend[colum1], embedding_friend[colum2], cmap=cmap, s=s)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontsize=fontsize)
    plt.savefig('../output/images/embedding_vectors.png')
    print("ploting tweet's vectors...")
    return plt.show()


def clustering(umap_embedding_fit, umap_embedding_predict, min_cluster_size, prediction_data):
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=prediction_data).fit(umap_embedding_fit)
    clustering = hdbscan.fit_predict(umap_embedding_predict)
    labels = hdbscan.labels_

    return clustering, labels


def plotClusterTogether(embedding_user, embedding_friend, colum1, colum2, c_user,
                        c_friend, s, fontzise, title):
    plt.figure(figsize=(15, 10))
    plt.scatter(embedding_user[colum1], embedding_user[colum2], c=c_user, s=s)
    plt.scatter(embedding_friend[colum1], embedding_friend[colum2], c=c_friend, s=s)
    plt.title(title, fontsize=fontzise)
    plt.savefig('../output/images/cluster_user_friend.png')
    print("plotting cluster from user and friend...")
    return plt.show()


def plotClusterUser(embedding_user, colum1, colum2, clustering_user, c_user, cmap, s, fontzise, title):
    plt.figure(figsize=(15, 10))
    plt.scatter(embedding_user[colum1], embedding_user[colum2], c=c_user, cmap=cmap, s=s)
    plt.colorbar(boundaries=len(np.unique(clustering_user)) - 0.5).set_ticks(len(np.unique(clustering_user) - 1))
    plt.title(title, fontsize=fontzise)
    plt.savefig('../output/images/cluster_user.png')
    print("plotting cluster from user...")
    return plt.show()


def plotClusterFriend(embedding_friend, colum1, colum2, clustering_friend, c_friend, cmap, s, fontzise, title):
    plt.figure(figsize=(15, 10))
    plt.scatter(embedding_friend[colum1], embedding_friend[colum2], c=c_friend, cmap=cmap, s=s)
    plt.colorbar(boundaries=len(np.unique(clustering_friend)) - 0.5).set_ticks(len(np.unique(clustering_friend) - 1))
    plt.title(title, fontsize=fontzise)
    plt.savefig('../output/images/cluster_user.png')
    print("plotting cluster from user...")
    return plt.show()


"Tweets'cluster from user"
'''


fig = plt.figure(figsize=(15, 10))
plt.scatter(umap_df['emb_1'], umap_df['emb_2'], c ='g', s=15)
plt.scatter(umap_df_friend['emb_1'],umap_df_friend['emb_2'], c='k', s=10)
plt.colorbar(boundaries=np.arange(7)-0.5).set_ticks(np.arange(6))



tfidf_vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer, min_df=3).fit(tweets_user['text'].values)

%time tfidf_matrix = tfidf_vectorizer.transform(tweets_user['text'].values)

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

array_matrix = tfidf_matrix.toarray()

first_vector_tfidfvectorizer=tfidf_matrix[0]
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False).head()

vocab = tfidf_vectorizer.vocabulary_

dist = 1 - cosine_similarity(tfidf_matrix)
dist

umap = UMAP(n_components=2, n_neighbors=8, random_state=42).fit(tfidf_matrix)
embedding = umap.fit_transform(tfidf_matrix)

# tweets from your colega
#embedding_colega = umap.transform(tfidf_matrix_colega)

umap_df = pd.DataFrame(embedding, columns=[f'emb_{i+1}' for i in range(2)])

umap_df.head()

plt.scatter(umap_df['emb_1'], umap_df['emb_2'], cmap='Spectral', s=8)
plt.gca().set_aspect('equal', 'datalim')
#plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection', fontsize=12);

hdbscan = HDBSCAN(min_cluster_size=15, prediction_data=True).fit(embedding)
clustering = hdbscan.fit_predict(embedding)
np.unique(clustering)

labels = hdbscan.labels_

fig = plt.figure(figsize=(15, 10))
plt.scatter(umap_df['emb_1'], umap_df['emb_2'], c =labels, cmap='RdYlBu', s=15)
plt.colorbar(boundaries=np.arange(7)-0.5).set_ticks(np.arange(6))
plt.title('UMAP projection Noelia', fontsize=12);
#plt.scatter(*embedding[outliers].T, s=50, linewidth=0, c='red', alpha=0.5)'''
