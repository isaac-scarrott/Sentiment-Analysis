# -*- coding: utf-8 -*-
import pandas as pd  
import time
import re
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class CleanTweet:
    WORD = re.compile(r'\w+')
    
    def __init__(self, tweetText):
        self.tempTweetText = str(tweetText)
        
    def removeUser(self):
        pattern = re.compile('@[A-Za-z0-9_]+')
        return pattern.sub('', self.tempTweetText)
    
    def removeRT(self):
        pattern = re.compile('RT : ')
        return pattern.sub('', self.tempTweetText)
    
    def removeURL(self):
        pattern = re.compile('((http(s)?:\/\/)|(www\.))(\S)*')
        return pattern.sub('', self.tempTweetText)
    
    def removePunctuation(self):
        self.tempTweetText = self.tempTweetText.translate(str.maketrans("", "", '\"\'!@£$%^&*()_+-=1234567890[];\\`~§±,./{}:|<>?#'))
        pattern = re.compile('\bamp\b')
        return pattern.sub('', self.tempTweetText)
    
    def removeNumbers(self):
        pattern = re.compile('\d+')
        return pattern.sub('', self.tempTweetText)

    
    def cleanTweetSequence(self):
        self.tempTweetText = self.removeUser()
        self.tempTweetText = self.removeRT()
        self.tempTweetText = self.removeURL()
        self.tempTweetText = self.removePunctuation()
        self.tempTweetText = self.removeNumbers()
        self.tempTweetText = self.tempTweetText.lower()

cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv(ROOT_DIR + "/Data/RawData/movieNegative.csv",header=None, names=cols, encoding='latin-1')
df.drop(['id','date','query_string','user'],axis=1,inplace=True)
#df = df.iloc[795000:805000, :]

temparray = []

start_time = time.time()

for index, row in df.iterrows():
    tempTweet = CleanTweet(row["text"])
    tempTweet.cleanTweetSequence() 
    temparray.append([row["sentiment"], tempTweet.tempTweetText])

newdf = pd.DataFrame(temparray)

print("--- %s seconds ---" % (time.time() - start_time))

newdf.to_csv("Data/CleanedData/movieNegative.csv", index=False)