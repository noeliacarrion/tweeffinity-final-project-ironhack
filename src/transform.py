import datetime
import re
from collections import Counter

import es_core_news_sm
import matplotlib.pyplot as plt
import pandas as pd
from spacy.lang.es.stop_words import STOP_WORDS
from wordcloud import WordCloud


def openData(file):
    print("opening dataframes...")
    dataframe = pd.read_csv(file)
    return dataframe


def rtweet(input_text):
    if input_text.startswith("RT"):
        value = 1
    else:
        value = 0
    return value


def mention(data, colum, regex):
    return data[colum].str.findall(regex).apply(','.join)


def hashtag(data, colum, regex):
    return data[colum].str.extract(regex, expand=False)


def createDate(dataframe, new_column):
    dataframe[new_column] = pd.to_datetime(dataframe[new_column], infer_datetime_format=True)
    return dataframe[new_column]


def dropcolumns(dataframe, columns):
    dataframe = dataframe.drop(columns, axis=1)
    return dataframe


def modifiedDate(dataframe, column, year, month, day):
    dataframe = dataframe[(dataframe[column] > datetime.date(year, month, day))]
    return dataframe


nlp = es_core_news_sm.load(parser=True)
nlp.Defaults.stop_words |= {"RT", "próx", "xd", "rt", "htt", "parir", "sobrar", "the", "and", "gracias", "hola",
                            "jajaja", "jajajaja", "hablar", "comer", "personar", "you", "with", "casar",
                            "was", "that", "what", "pasar", "salir"}


def spacyTokenizer(sentence):
    sentence = re.sub(r"htt\S+", '', sentence)
    sentence = re.sub('@', '', sentence)
    tokens = nlp(sentence)
    filtered_tokens = []
    for word in tokens:
        lemma = word.lemma_.lower().strip()
        if lemma not in STOP_WORDS and re.search("^[a-zA-Z]{2}\w+", lemma):
            filtered_tokens.append(lemma)
    return filtered_tokens


def dropRows(data, colum):
    return data[data[colum].map(lambda x: len(x)) > 0]


def wordCloud(user, series):
    print("we are preparing yours and your friends's wordcloud...")
    all_words = []
    for line in series:
        all_words.extend(line)

    wordfreq = Counter(all_words)
    wordcloud = WordCloud(width=900, height=600, max_words=500, max_font_size=100, relative_scaling=0.5,
                          colormap='Blues',
                          normalize_plurals=True).generate_from_frequencies(wordfreq)
    plt.figure(figsize=(12, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('../output/%s_wordcloud.png' % user)
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')
