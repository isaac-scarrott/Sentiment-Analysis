#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 00:48:34 2019

@author: isaacscarrott
"""
import numpy as np
import pandas as pd
from pickle import load
import gensim
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

print("Loading in data....")
cols = ['sentiment','text']
trainDF = pd.read_csv(ROOT_DIR + "/Data/CleanedData/trainingData.csv",header=None, names=cols, encoding='latin-1')
trainDF = trainDF.iloc[1:]
trainDF = trainDF.dropna()

testDF = pd.read_csv(ROOT_DIR + "/Data/CleanedData/testingdata.csv",header=None, names=cols, encoding='latin-1')
testDF = testDF.iloc[1:]
testDF = testDF.dropna()

train_x = [row.split() for row in trainDF['text']]
test_x = [row.split() for row in testDF['text']]

del trainDF, testDF,cols 

tweet_w2v = gensim.models.Word2Vec.load(ROOT_DIR + "/Models/word2vec.model")
dimension = 200
#tweet_w2v = gensim.models.KeyedVectors.load_word2vec_format('/Users/isaacscarrott/Downloads/GoogleNews-vectors-negative300.bin', binary=True) 
#dimension = 300
with open(ROOT_DIR + "/Models/TFIDF.pickle", 'rb') as handle:
    tfidf = load(handle)

print("Building Tweet Vectors....")
train_vecs_w2v = np.concatenate([buildWordVector(z, dimension) for z in train_x])

print("Building Tweet Vectors....")
test_vecs_w2v = np.concatenate([buildWordVector(z, dimension) for z in test_x])

np.save("Data/VectorsData/trainingTweetVectors.npy", train_vecs_w2v)
np.save("Data/VectorsData/testingTweetVectors.npy", test_vecs_w2v)