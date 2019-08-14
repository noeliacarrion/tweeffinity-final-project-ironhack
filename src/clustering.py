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


'''
 def plotEmbedding(embedding_user, embedding_friend, colum1, colum2, cmap, s, title, fontsize):
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding_user[colum1], embedding_user[colum2], cmap=cmap, s=s)
    plt.scatter(embedding_friend[colum1], embedding_friend[colum2], cmap=cmap, s=s)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(title, fontsize=fontsize)
    plt.savefig('../output/images/embedding_vectors.png')
    print("plotting tweet's vectors...")
    return plt.show()
'''


def clustering(umap_embedding_fit, umap_embedding_predict, min_cluster_size, prediction_data):
    print("clustering...")
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=prediction_data).fit(umap_embedding_fit)
    clustering = hdbscan.fit_predict(umap_embedding_predict)
    labels = hdbscan.labels_

    return clustering, labels


def plotClusterUser(embedding_user, colum1, colum2, clustering_user, c_user, cmap, s, fontzise, title):
    print("please, be patient... Thank you")
    print("plotting cluster from you...")
    plt.figure(figsize=(15, 10))
    plt.scatter(embedding_user[colum1], embedding_user[colum2], c=c_user, cmap=cmap, s=s)
    plt.colorbar(boundaries=np.unique(clustering_user) - 0.5).set_ticks(np.unique(clustering_user) - 1)
    plt.title(title, fontsize=fontzise)
    plt.savefig('../output/cluster_user.png')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')


def plotClusterTogether(embedding_user, embedding_friend, colum1, colum2, c_user,
                        c_friend, s, fontzise, title):
    print("plotting your cluster and your friend's tweets")
    plt.figure(figsize=(15, 10))
    plt.scatter(embedding_user[colum1], embedding_user[colum2], c=c_user, s=s)
    plt.scatter(embedding_friend[colum1], embedding_friend[colum2], c=c_friend, s=s)
    plt.title(title, fontsize=fontzise)
    plt.savefig('../output/cluster_user_friend.png')
    plt.show()
    plt.pause(2)
    plt.close('all')


'''
def plotClusterFriend(embedding_friend, colum1, colum2, clustering_friend, c_friend, cmap, s, fontzise, title):
    print('plotting cluster from your friend...')
    plt.figure(figsize=(15, 10))
    plt.scatter(embedding_friend[colum1], embedding_friend[colum2], c=c_friend, cmap=cmap, s=s)
    plt.colorbar(boundaries=np.unique(clustering_friend) - 0.5).set_ticks(np.unique(clustering_friend) - 1)
    plt.title(title, fontsize=fontzise)
    plt.savefig('../output/images/cluster_friend.png')
    return plt.show()
'''
