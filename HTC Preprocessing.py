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
df = pd.read_csv('C:\\Users\\Ahmed\\PycharmProjects\\SA\\tweetList.txt', sep='deliminator', quoting=csv.QUOTE_NONE, error_bad_lines=False,names=['c'+str(i) for i in range(1000)])
df=df.dropna(how='all', axis=1)
df.shape


df['no_of_keys'] = np.nan
df['keys'] = np.nan
lst = []
for i in range(len(df)):
    r = str(df.ix[i, 'c0'])
    # response_item = ast.literal_eval(json.dumps(r, ensure_ascii=False).encode('utf8'))
    try:
        d = json.loads(r)
        df.ix[i, 'no_of_keys'] = len(d.keys())
        df.ix[i, 'keys'] = str(list(d.keys()))
        lst = list(set(lst + list(d.keys())))
        len(lst)
    except:
        pass
df.info()
df = df[~isNaN(df['no_of_keys'])]
df.shape
df['no_of_keys'].value_counts()
df['keys'].value_counts()
len(list(pd.unique(df['keys'])))
for c in lst:
    df[c] = np.nan
df.info()
df = df.reset_index(drop=True)
df.info()
for i in range(len(df)):
    print(i)
    r = str(df.ix[i, 'c0'])
    d = json.loads(r)
    for c in lst:
        try:
            df.ix[i, c] = str(d.get(c, None))
        except:
            df.ix[i, c] = str(d.get(c, None).encode('utf-8'))

df.info()

df.to_csv("input1.csv", index=False)
df = pd.read_csv("input1.csv")
df.shape
df['quoted_status'].value_counts()
ri = df[df['quoted_status'] != 'None'].index.values.tolist()
len(ri)
df['quoted_status_text'] = np.nan
for i in ri:
    print(i)
    r = str(df.ix[i, 'quoted_status'])
    d = r.split("\'text\': ")[1].split(", u\'is_quote_status\':")[0]
    df.ix[i, 'quoted_status_text'] = d.encode('utf-8').replace("'", '').replace('"', '')[1:]
df['quoted_status_text'].value_counts()
df.to_csv("input_after_cleaning.csv", index=False)

df = pd.read_csv("input_after_cleaning.csv")
df.shape
# (22177, 34)
len(pd.unique(df['id']))
# 17530
df = df.drop_duplicates()
df.shape
# (21182, 34)
df = df[['id', 'text', 'quoted_status_text']]
df = df.drop_duplicates()
df.shape
df['id'].value_counts()
df.to_csv("input_to_consider.csv", index=False)
df.info()

df = pd.read_csv("input_to_consider.csv")
df.info()
df = df.head(15)
df['text1'] = df['text']
df['text1'] = np.where(~isNaN(df['quoted_status_text']), df['text'] + ' ' + df['quoted_status_text'], df['text1'])
df['text1'].value_counts()
df['text2'] = df['text1']
c = 'text2'
df[c] = map(str, df[c])
df[c] = df[c].replace(to_replace='\n', value=' ', regex=True)
df[c] = df[c].replace(to_replace='\r', value=' ', regex=True)
df[c] = df[c].replace(to_replace='\t', value=' ', regex=True)
df[c] = df[c].str.title()
df[c] = df.apply(lambda x: " ".join(x[c].split()), axis=1)
df[c] = df.apply(lambda x: re.sub(r'Https:\S+', '', str(x[c])), axis=1)
df[c] = df.apply(lambda x: (" ".join(re.findall("[a-zA-Z0-9'@#]+", x[c]))).lower(), axis=1)
df[c].value_counts()
len(pd.unique(df[c]))
df['text2'].value_counts()
df[df.apply(lambda x: not 'htc' in x['text2'], axis=1)]
content = list(df['text2'])

cached = stopwords.words("english")
cached.append('amp')
cached = list(set(cached))
len(cached)
i = 0
cleaned_content = []
allwords = []
while i < len(content):
    print i
    words = re.findall('[a-zA-Z]+-*[a-zA-Z]*', content[i])
    words = [word for word in words if word.lower() not in cached]
    words = [word for word in words if len(word) > 2]
    words = [WordNetLemmatizer().lemmatize(word, 'v') for word in words]
    #words = [singularize(word) for word in words]
    words = [word for word in words if word.lower() not in cached]
    words = [word for word in words if len(word) > 2]
    cleaned_content.append(' '.join(list(set(words))))
    allwords = allwords + words
    i += 1

allwords = list(set(allwords))
len(allwords)  # 8966
'an' in allwords
len(cleaned_content)  # 17530
allwords[:10]
df['text3'] = cleaned_content
df.to_csv("Part_1.csv", index=False)