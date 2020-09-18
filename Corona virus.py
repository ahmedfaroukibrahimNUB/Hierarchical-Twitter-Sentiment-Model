# import Required Model
from scipy.cluster.hierarchy import ward, dendrogram
from matplotlib import pyplot as plt
import scipy
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
stopwords = nltk.corpus.stopwords.words('english')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
# from nltk.corpus import stopwords

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score.values()
################################
# Extract Actual data work on it
################################
df= pd.read_csv("C:\\Users\\Ahmed\\PycharmProjects\\SA\\corona_final_train_data2.csv")
df=df.drop_duplicates('text3')
df.info()
#################################################
############## VADER Sentiment Analysis ########
################################################
df['neg'] = np.nan
df['neu'] = np.nan
df['pos'] = np.nan
df['compound'] = np.nan
for i in range(len(df)):
    vv = sentiment_analyzer_scores(str(df.ix[i, 'text3']))
    df.ix[i, 'neg'] = vv[0]
    df.ix[i, 'neu'] = vv[1]
    df.ix[i, 'pos'] = vv[2]
    df.ix[i, 'compound'] = vv[3]
    #print(i)
df['pn']='neu'
df['pn'] = np.where(df['compound'] >= 0.05, 'p', df['pn'])
df['pn'] = np.where(df['compound'] <= -0.05, 'n', df['pn'])
polarity_count=df['pn'].value_counts()
print 'the polarity is catogerizes as'
print polarity_count
#ids = list(df['id'])
#################################################
# Before proceeding to extract the concepts, let us visualize the tweets with the help of a #wordcloud
#################################################
cleaned_content = list(df['text3'])
from wordcloud import WordCloud
cleaned_content = list(df['text3'])
comment_words = ' '.join(cleaned_content)
wordcloud = WordCloud().generate(comment_words)
# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

################################################
# create hierarchical tree
################################################
synopses = list(df['text3'])
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in synopses:
    allwords_stemmed = tokenize_and_stem(i)  # for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)
# print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'
# there are 99535 items in vocab_frame
# print vocab_frame.head()
from sklearn.feature_extraction.text import TfidfVectorizer

# define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)  # fit the vectorizer to synopses

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)

from scipy.cluster.hierarchy import ward, dendrogram
from matplotlib import pyplot as plt
import scipy
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt

linkage_matrix = ward(dist)

PP = 6
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=PP,  # show only the last p merged clusters
    show_leaf_counts=True,  # otherwise numbers in brackets are counts
    #color_threshold=27,
    # orientation='right',
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True  # to get a distribution impression in truncated branches
)
#plt.axhline(y=27, c='yellow', lw=1, linestyle='dashed')
plt.show()
plt.savefig('ward_clusters.png', dpi=200)
plt.close()
######################################################################
#This is the distribution of tweets in clusters
########################################################################
from collections import Counter

from scipy.cluster.hierarchy import fcluster

max_d = 50
clusters = fcluster(linkage_matrix, PP, criterion='maxclust')
# pp is max number of clusters requested
df['Corona_clusters'] = clusters


def CountFrequency(my_list):
    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    for key, value in freq.items():
        print ("% d : % d" % (key, value))


CountFrequency(list(clusters))
df['Corona_clusters'] = clusters
df.to_csv('corona_clusters.csv', index=False)   # to determine distribution of  each tweets in clusters from 1 to 6
##############################################
# The output is showing the distribution of whole data (12520) into 6 leaves - first leave (node # 2) has 405 messages
##############################################
################################################
#Extractions of aspects
################################################


def reemovNestings(l):
    for i in l:
        if type(i) == list:
            reemovNestings(i)
        else:
            cls.append(i)


import operator


def top5(cls1):
    df1 = df[df['Corona_clusters'].isin(cls1)]
    D = list(df1['text3'])
    D = [' '.join(list(set(x.split()))) for x in D]
    txt = ' '.join(D)
    split_it = txt.split()
    counter = Counter(split_it)
    most_occur = counter.most_common(5)
    d = dict(most_occur)
    l = list(d.values())
    k = list(d.keys())
    indx = (sorted(range(len(l)), key=lambda k: l[k]))
    indx.reverse()
    k = [k[i] for i in indx]
    return ' '.join(k)


# T=[1, [2, 3]]
def recur(T):
    global cls
    global topic
    for TT in T:
        print(TT)
        if type(TT) != list:
            TT = [TT]
        cls = []
        reemovNestings(TT)
        topic.append(top5(cls))
        print (topic)
        if (any(isinstance(i, list) for i in TT)):
            recur(TT)
        if len(TT) != 1:
            recur(TT)

topic = []
T = [[1, 2], [3, [4, [5, 6]]]]
# T=[[1, 2], [3, [4, [5, [6, [7, 8]]]]]]
topic = []
cls = []
reemovNestings(T)
topic.append(top5(cls))
print(T)
print (topic)
recur(T)
############################################
############################################
############################################
def pscore(cls1):
    df1 = df[df['Corona_clusters'].isin(cls1)]
    p = len(df1[df1['pn'] == 'p'])
    n = len(df1[df1['pn'] == 'n'])
    return (float(p) / (p + n))


def precur(T):
    global cls
    global topic
    global lsst
    for TT in T:
        print(TT)
        if type(TT) != list:
            TT = [TT]
        cls = []
        reemovNestings(TT)
        lsst.append(TT)
        o.append(pscore(cls))
        print (o)
        if (any(isinstance(i, list) for i in TT)):
            precur(TT)
        if len(TT) != 1:
            precur(TT)


o = []
cls = []
lsst = []
reemovNestings(T)
o.append(pscore(cls))
lsst.append(T)
print(T)
print (o)
precur(T)

len(o)
len(topic)

########################################
# Let us see the scores of the tweets
########################################
dff = pd.DataFrame(topic)
dff.columns = ['topic']
dff['score'] = o
dff['node'] = lsst
dff.info()
dff=dff.drop_duplicates('score')
dff=dff[['node','topic','score']]
dff.to_csv('Corona_Polarity_Score.csv', index=False)
df.to_csv('Corona_Sentiment_Polarity&Clusters.csv', index=False)