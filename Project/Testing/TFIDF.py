# -*- coding: utf-8 -*-

from pandas import read_csv
from pickle import dump, HIGHEST_PROTOCOL
from sklearn.feature_extraction.text import TfidfVectorizer
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Importing CSV File...")
cols = ['sentiment','text']
df = read_csv(ROOT_DIR + "/Data/CleanedData/trainingdata.csv",header=None, names=cols, encoding='latin-1')
df = df.iloc[1:]
train_x = [row.split() for row in df['text']]
train_y = list(df['sentiment'])
del df, cols


print("Building tf-idf matrix ...")
vectorizer = TfidfVectorizer(analyzer=lambda x: x)
matrix = vectorizer.fit_transform(train_x)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

print("Saving tf-idf matrix ...")
with open('Models/TFIDF.pickle', 'wb') as handle:
    dump(tfidf, handle, protocol=HIGHEST_PROTOCOL)
