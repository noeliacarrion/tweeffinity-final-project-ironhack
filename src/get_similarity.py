from clustering import *


def addColum(dataframe, new_column, clustering_values):
    dataframe[new_column] = clustering_values
    return dataframe[new_column]


def mergeDataframe(dataframe1, dataframe2):
    dataframe_merge = pd.concat([dataframe1, dataframe2], axis=1)
    return dataframe_merge


def compareSimilarity(data_user, data_friend, colum1, colum2, colum3):
    print("obtaining similarity between you and you friend")
    data_friend_shared = data_friend[
        (data_user[colum1].max() > data_user[colum1]) & (data_user[colum1].min() < data_friend[colum1]) & (
                data_friend[colum2] < data_user[colum2].max()) & (
                data_user[colum2].min() < data_friend[colum2])]

    data_friend_shared = data_friend_shared[data_friend_shared[colum3] != -1]

    result = round(100 * (len(data_friend_shared) / len(data_friend)), 2)
    if result < 20:
        similarity = 'Oh! We are not good friends. We just have an affinity of {}%'.format(result)
    else:
        similarity = 'Amazing! We are still friends. We have an affinity of {}%'.format(result)

    return similarity

