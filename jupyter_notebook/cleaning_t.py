import pandas as pd
import re
import numpy as np
from time import time 
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.cluster import KMeans
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
%pylab inline

def open_data(file):
    df = pd.read_csv(file)
    return df

tweets_user = open_data('/Users/Noelia/Desktop/DATA_IRONHACK/FINAL_PROJECT/final_project_ironhack/output/User_csv/@mew4teaching_tweets.csv')

#tweets_user = pd.read_csv('/Users/Noelia/Desktop/DATA_IRONHACK/FINAL_PROJECT/final_project_ironhack/output/User_csv/@mew4teaching_tweets.csv')

tweets_user.head()

#tweet_text = tweets_user[['text']]

def retweet(input_text):
    if input_text.startswith("RT"):
        value = 1
    else:
        value = 0
    return value

tweets_user['RT'] = tweets_user.text.apply(retweet)

tweets_user['mention'] = tweets_user.text.str.findall(r'(?<![@\w])@(\w{1,25})').apply(', '.join)

tweets_user['hashtag'] = tweets_user.text.str.extract(r'(\#\w+)', expand=False)

tweets_user['hashtag'] = tweets_user['hashtag'].fillna("")

#def remove_characters(input_text):
#    list_remove = ["RT", "(?<![@\w])@(\w{1,25})", "\#+"]
#    for charac in list_remove:
#        input_text = re.sub(charac, '', input_text)
#    return input_text.lower()
#def remove_special_characters(data, colum):
#    input_text = data[colum].str.replace("[\d\W_]+", " ", re.UNICODE)
#    return input_text
#def deEmojify(inputString):
 #   return inputString.encode('latin-1', 'ignore').decode('latin-1')#

#def removeSpace(x):
 #       return " ".join([w for w in x.split() if len(w)>3])

#tweets_user['cleaned_tweets'] = tweets_user.text.apply(remove_characters)

#tweets_user[['cleaned_tweets']].head()

#tweets_user['cleaned_tweets'] = tweets_user['cleaned_tweets'].str.replace('http\S+|www.\S+', "")

#tweets_user['cleaned_tweets'] = remove_special_characters(tweets_user, "cleaned_tweets")

#tweets_user['cleaned_tweets'] = tweets_user['cleaned_tweets'].apply(deEmojify)

#tweets_user['cleaned_tweets'] = tweets_user['cleaned_tweets'].apply(removeSpace)

#tweets_user['cleaned_tweets']

#def remove_characters(input_text):
 #   list_remove = ["RT", "@", "\#+", 'http\S+|www.\S+']
  #  for charac in list_remove:
   #     input_text = re.sub(charac, '', input_text)
   # return input_text.lower()

#tweets_user['tokenized'] = tweets_user['text'].apply(spacy_tokenizer)

#tweets_user = tweets_user[tweets_user['tokenized'].map(lambda d: len(d)) > 0]

#tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

#vector2 = bow_vector.fit_transform(all_tweets)

tweets_user['date'] = tweets_user['created_at']

tweets_user['date'] = pd.DatetimeIndex(tweets_user['created_at']).to_period('D')

tweets_user = tweets_user.drop(columns=["created_at"], axis=1)

#tweets_user = tweets_user[["id", "date", "mention", "hashtag", "RT", "cleaned_tweets"]]

#def tokenize(s):
    #return s.split(" ") 

nlp = spacy.load('en')

# English tokenizer, tagger, parser, NER and word vectors
parser = English()

nlp.Defaults.stop_words |= {"rt","amp","xx", "xxx"}

#nlp.Defaults.stop_words.add([["rt", "amp"]])

def spacy_tokenizer(sentence):
    #sentence = remove_characters(sentence) 
   # sentence = sentence.replace('http\S+|www.\S+', "") 
    tokens = parser(sentence)
    filtered_tokens = []
    for word in tokens:
        lemma = word.lemma_.lower().strip()
        
        if lemma not in STOP_WORDS and re.search('^[a-zA-Z]{2}', lemma):
            filtered_tokens.append(lemma)

    return filtered_tokens

#" ".join([token.lemma_ for token in doc])


#if word.lemma_ != "-PRON-" else word.lower_

tweets_user['tokenized'] = tweets_user['text'].apply(spacy_tokenizer)

num_features = 300
min_word_count = 50
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-4
seed = 2

all_words = []
for line in tweets_user['tokenized']: # try 'tokens'
    all_words.extend(line)
    
# create a word frequency dictionary
wordfreq = Counter(all_words)
# draw a Word Cloud with word frequencies
wordcloud = WordCloud(width=900,
                      height=500,
                      max_words=500,
                      max_font_size=100,
                      relative_scaling=0.5,
                      colormap='Blues',
                      normalize_plurals=True).generate_from_frequencies(wordfreq)
plt.figure(figsize=(17,14))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#def tokenize_only(text):
 #   tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
  #  filtered_tokens = []
    #filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
   # for token in tokens:
  #      if re.search('[a-zA-Z]', token):
    #        filtered_tokens.append(token)
   # return filtered_tokens

#totalvocab_stemmed = []
#totalvocab_tokenized = []
#for i in tweets_user['text']:
 #   allwords_stemmed = tokenize_and_stem(i)
  #  totalvocab_stemmed.extend(allwords_stemmed)
    
   # allwords_tokenized = tokenize_only(i)
   # totalvocab_tokenized.extend(allwords_tokenized)

#vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
#print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer, min_df=3)

%time tfidf_matrix = tfidf_vectorizer.fit_transform(tweets_user['text'].values)

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

array_matrix = tfidf_matrix.toarray()

first_vector_tfidfvectorizer=tfidf_matrix[0]
df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
df.sort_values(by=["tfidf"],ascending=False).head()

vocab = tfidf_vectorizer.vocabulary_

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
dist

from mpl_toolkits.mplot3d import Axes3D
import umap

umap = UMAP(n_components=2, n_neighbors=8)
embedding = umap.fit_transform(dist)

umap_df = pd.DataFrame(embedding, columns=[f'emb_{i+1}' for i in range(2)])

umap_df.head()

plt.scatter(umap_df['emb_1'], umap_df['emb_2'], cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection', fontsize=12);

from hdbscan import HDBSCAN
    
hdbscan = HDBSCAN(min_cluster_size=30, gen_min_span_tree=True, metric='braycurtis')

clustering = hdbscan.fit_predict(embedding)

np.unique(clustering)

hdbscan.labels_.max()

labels = hdbscan.labels_
labels

fig = plt.figure(figsize=(15, 10))
plt.scatter(embedding[:,0], embedding[:,1], c=clustering);
plt.colorbar(boundaries=np.arange(12)-0.5).set_ticks(np.arange(11))

from sklearn.metrics import silhouette_score

print("Silhouette Coefficient: %0.3f"
      % silhouette_score(embedding, hdbscan.fit_predict(embedding)))

from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans

num_clusters = 9

km = KMeans(n_clusters=num_clusters)

%time km.fit(embedding)

clusters = km.labels_.tolist()

from matplotlib.pyplot import figure
fig = plt.figure(figsize=(15, 10))
plt.scatter(umap_df['emb_1'], umap_df['emb_2'], c =km.labels_, cmap='Spectral')
#plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(5)-0.5).set_ticks(np.arange(4))
centers = np.array(km.cluster_centers_)

#This array is one dimensional, thus we plot it using:
plt.scatter(centers[:,0], centers[:,1], marker="x", color='green')
plt.title('UMAP projection', fontsize=12);

from sklearn.metrics import silhouette_score

print("Silhouette Coefficient: %0.3f"
      % silhouette_score(embedding, km.predict(embedding)))

inertia = []

for n_clusters in range(2, 12):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embedding)
    inertia.append(kmeans.inertia_)
    
plt.plot(range(2, 12), inertia);

tweets = {'tweet': tweets_user['text'].values,  'cluster': clusters}
frame = pd.DataFrame(tweets, index = [clusters] , columns = ['tweet', 'cluster'])

frame[frame['cluster']==1].head()

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)
tfidf_df.head()

def get_df_from_cluster(cluster):
    return tfidf_df[clustering==cluster]

top_words_cluster = get_df_from_cluster(5).T.sum(axis=1).sort_values(ascending=False)
top_words_cluster.head()
