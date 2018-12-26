import re, string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from textblob import TextBlob


class CleanTweet:
    def __init__(self, tweetText):
        self.tempTweetText = tweetText
        self.tempTweetListFilterd = []
        self.weight = []
        
    def removeUser(self):
        pattern = re.compile('\w*@\w*')
        return pattern.sub('', self.tempTweetText)
    
    def removeRT(self):
        pattern = re.compile('RT : ')
        return pattern.sub('', self.tempTweetText)
    
    def removeURL(self):
        pattern = re.compile('http\S+')
        return pattern.sub('', self.tempTweetText)
    
    def removeStopWords(self):
        stop_words = set(stopwords.words('english')) 
        
        tempTweetList = word_tokenize(self.tempTweetText)
        
        self.tempTweetListFilterd = [w for w in tempTweetList if not w in stop_words]
        self.tempTweetListFilterd = []
        
        for w in tempTweetList:
            if w not in stop_words:
                self.tempTweetListFilterd.append(w)
        return
    
    def removePunctuation(self):
        for index, word in enumerate(self.tempTweetListFilterd):
            self.tempTweetListFilterd[index] = re.sub(r'[^\w\s]','',word)
            self.tempTweetListFilterd[index] = re.sub('^[0-9]+', '', self.tempTweetListFilterd[index])
            if self.tempTweetListFilterd[index] == '':
                del self.tempTweetListFilterd[index]
                del self.weight[index]
        return
    
    def decapitalize(self):
        for index, word in enumerate(self.tempTweetListFilterd):
            self.tempTweetListFilterd[index] = word.lower()
        return 
    
    def addWeight(self):
        for index, word in enumerate(self.tempTweetListFilterd):
            if len(self.weight) == index:
                if word.startswith('#'):
                    self.weight.append(1.5)
                    self.weight.append(1.5)
                elif word.isupper():
                    self.weight.append(1.5)
                else:
                    self.weight.append(1)
        return
    
    def correctSpelling(self):
        self.tempTweetText = str(TextBlob(self.tempTweetText).correct())
            
temp = CleanTweet("RT : @Isaac  #Banana isn't is a very good string to test https://www.google.co.uk/")

temp.tempTweetText = temp.removeUser()
temp.tempTweetText = temp.removeRT()
temp.tempTweetText = temp.removeURL()

temp.correctSpelling()
temp.removeStopWords()
temp.addWeight()
temp.decapitalize()
temp.removePunctuation()

print(temp.tempTweetListFilterd)
print(temp.weight)



