import gensim
from keras.backend import clear_session
import tensorflow as tf

import tweepy as tw
import nltk
from nltk.corpus import stopwords 

from pickle import load
import numpy as np

import re
import os

from keras.models import model_from_json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class sentimentAnalysisObject:
  
    def __init__(self):
        clear_session()
        auth = tw.OAuthHandler('gtnUtDlSZBrWTr2IpELWfa5UV', '477YfswV7vunwngWztB59lDAFKIcN2Octd95a4jeD4dHVDLS9e')
        auth.set_access_token('373424309-K0tLnnKPnO6yieEjd1CmnLbSR40Ck5TbFNJMFAay', 'Q9QiFKtY3KYZ0DZXUT1x0kj0gxQs8rPsQ5E8ZDwPwqX7Z')
        self.api = tw.API(auth, wait_on_rate_limit=True)
        self.number = 200
        self.tweet_w2v = gensim.models.Word2Vec.load(ROOT_DIR + "/Models/word2vec.model")
        self.alltweets = ""
        
        with open(ROOT_DIR + "/Models/TFIDF.pickle", 'rb') as handle:
            self.tfidf = load(handle)
        # load json and create model
        json_file = open('Models/SentimentNN.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        self.loaded_model.load_weights("Models/SentimentNN.h5")
        
        self.loaded_model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        global graph
        graph = tf.get_default_graph()
        
        trends = self.api.trends_place(23424975)
        trends = ([trend['name'] for trend in trends[0]['trends']])
        trends = trends[0:5]
        
    def getCurrentTrends(self):
        trends = self.api.trends_place(23424975)
        trends = ([trend['name'] for trend in trends[0]['trends']])
        trends = trends[0:5]
        
        return str(trends[0]), str(trends[1]), str(trends[2]), str(trends[3]), str(trends[4])
        
        
    def CleanTweet(self, tweet):
        patternArary = ['@[A-Za-z0-9_]+', 'RT : ', '((http(s)?:\/\/)|(www\.))(\S)*', '\d+', '\bamp\b']
        
        for current in patternArary:
            pattern = re.compile(current)
            tweet = pattern.sub('', tweet)
            
        tweet = tweet.translate(str.maketrans("", "", '\"\'!@£$%^&*()_+-=1234567890[];\\`~§±,./{}:|<>?#'))
        tweet = tweet.encode('ascii', 'ignore').decode('ascii')  
                                      
        return tweet.lower()
    
    def buildWordVector(self, tokens, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0
        tokens = tokens.split()
        for word in tokens:
            try:
                vec += self.tweet_w2v[word].reshape((1, size)) * self.tfidf[word]
                count += 1.
            except KeyError: # handling the case where the token is not
                             # in the corpus. useful for testing.
                continue
        if count != 0:
            vec /= count
        return vec
        
    def sentimentSearchInput(self, keyword):
        
        search = tw.Cursor(self.api.search, q=keyword, result_type="mixed", lang="en", tweet_mode="extended").items(self.number)
        
        tweets = []
        uncleanedTweets = []
        for x in search:
            uncleanedTweets.append(x.full_text)
            tweets.append(self.CleanTweet(x.full_text))
            
        print("Building Tweet Vectors....")
        train_vecs_w2v = np.concatenate([self.buildWordVector(z, 200) for z in tweets])
        with graph.as_default():
            predictions = self.loaded_model.predict(train_vecs_w2v, batch_size=64, verbose=1)
        predictions2 = np.round(predictions)
        prediction = np.mean(predictions, axis=0)
        prediction2 = np.mean(predictions2, axis=0)
        
        for x in tweets:
            self.alltweets = self.alltweets + x
        
        return str((prediction+prediction2)/2*100)[1:5] + "%"
    
    def getNewTrends(self):
        stop_words = set(stopwords.words('english')) 
        
        tokens = nltk.word_tokenize(self.alltweets)
        
        filtered_sentence = [w for w in tokens if not w in stop_words] 
        
        bgs = nltk.bigrams(filtered_sentence)
        
        fdist = nltk.FreqDist(filtered_sentence)
        fdist = (fdist + nltk.FreqDist(bgs))
        
        trends = fdist.most_common(5)
        
        for i,x in enumerate(trends):
            trends[i] = str(x[0])
        
        return trends[0], trends[1], trends[2], trends[3], trends[4]
        
test = sentimentAnalysisObject()

test.sentimentSearchInput("death")
            
        
