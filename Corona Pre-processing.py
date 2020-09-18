import scipy.stats as ss
import glob, os
import pandas as pd
import numpy as np
import operator
from time import gmtime, strftime
from dateutil.parser import parse
import re
import math
import pandas as pd
from gensim import corpora, models, similarities
import re
import string
import numpy as np
import os
from datetime import datetime
from collections import Counter
import logging
from nltk.corpus import stopwords
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from pandas import *
#from pattern.text.en import singularize
import math
from scipy.stats import rankdata
import csv
import ast
import json, ast
from nltk.stem.wordnet import WordNetLemmatizer

def string_found(string1, string2):
    if isNaN(string2): return False
    if string2=='nan': return False
    if string1=='nan': return False
    if len(string1)==1: return False
    if re.search(r"\b" + re.escape(string1) + r"\b", string2):
        return True
    return False

def isNaN(num):
    return num != num

def union(a, b):
    return list(set(a) | set(b))

def intersect(a, b):
    return list(set(a) & set(b))

#read and preview raw data
#df= pd.read_csv("C:\\Users\\Ahmed\\PycharmProjects\\SA\\corona.csv")
df= pd.read_csv("C:\\Users\\Ahmed\\PycharmProjects\\SA\\corona.csv")
#df=df.dropna(how='all', axis=1)
df.shape
df.info()
#########
# print len(pd.unique(df['text3']))
# df=df['text3']
# df=df.drop_duplicates()
# #print df.shape

##############
#df['text1']=df['text']
#df['text1']=np.where(~isNaN(df['quoted_status_text']),df['text'] + ' ' + df['quoted_status_text'],df['text1'])
#df['te#concatenate main text and quoted text#concatenate main text and quoted textxt1'].value_counts()
df['text2']=df['text']
#df['text2']=df['text3']
c='text2'
df[c]=map(str, df[c])
df[c] = df[c].replace(to_replace='\n', value=' ', regex=True)
df[c] = df[c].replace(to_replace='\r', value=' ', regex=True)
df[c] = df[c].replace(to_replace='\t', value=' ', regex=True)
df[c] =df[c].str.title()
df[c]=df.apply(lambda x : " ".join(x[c].split()), axis=1)
df[c]=df.apply(lambda x :re.sub(r'Https:\S+', '', str(x[c])), axis=1)
df[c] = df.apply(lambda x : (" ".join(re.findall("[a-zA-Z0-9'@#]+",x[c]))).lower(), axis=1)
df[c].value_counts()
len(pd.unique(df[c]))

#######3
content = list(df['text2'])
cached = stopwords.words("english")
cached.append('amp')
cached=list(set(cached))
len(cached)
i = 0
cleaned_content = []
allwords = []
while i < len(content):
    #print i
    words = re.findall('[a-zA-Z]+-*[a-zA-Z]*' , content[i])
    words = [word for word in words if word.lower()  not in cached]
    words = [word for word in words if len(word)>2]
    words = [WordNetLemmatizer().lemmatize(word,'v') for word in words]
    #words = [singularize(word) for word in words]
    words = [word for word in words if word.lower()  not in cached]
    words = [word for word in words if len(word)>2]
    cleaned_content.append(' '.join(list(set(words))))
    allwords = allwords + words
    i += 1

# df['text4']=cleaned_content
# df['text4'].head()
#
# df.to_csv("C:\\Users\\Ahmed\\PycharmProjects\\SA\\corona_final_train_data.csv",index=False)
#
# df1=df.drop_duplicates('text4')
# df1.to_csv("C:\\Users\\Ahmed\\PycharmProjects\\SA\\corona_final_train_data2.csv",index=False)

df['text3']=cleaned_content
df['text3'].head()

df.to_csv("C:\\Users\\Ahmed\\PycharmProjects\\SA\\corona_final_train_data2.csv",index=False)

df1=df.drop_duplicates('text3')
df1.to_csv("C:\\Users\\Ahmed\\PycharmProjects\\SA\\corona_final_train_data2.csv",index=False)