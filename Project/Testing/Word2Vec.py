#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 00:20:10 2019

@author: isaacscarrott
"""
from pandas import read_csv
from gensim.models import Word2Vec
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Importing CSV File...")
cols = ['sentiment','text']
df = read_csv(ROOT_DIR + "/Data/CleanedData/trainingdata.csv",header=None, names=cols, encoding='latin-1')
df = df.iloc[1:]
train_x = [row.split() for row in df['text']]
train_y = list(df['sentiment'])
del df

print("Building model...")

model = Word2Vec(sentences=train_x, sg=1, size=200, window=6, workers=2, min_count=1)

print("Saving model ...")
model.save("Models/word2vec.model")
