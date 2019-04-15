import numpy as np
import pandas as pd
import gensim
from pickle import load
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading in data....")
cols = ['sentiment','text']
trainDF = pd.read_csv(ROOT_DIR + "/Data/CleanedData/trainingData.csv",header=None, names=cols, encoding='latin-1')
trainDF = trainDF.iloc[1:]
trainDF = trainDF.dropna()

testDF = pd.read_csv(ROOT_DIR + "/Data/CleanedData/testingdata.csv",header=None, names=cols, encoding='latin-1')
testDF = testDF.iloc[1:]
testDF = testDF.dropna()

train_x = np.asarray(trainDF['text'])
test_x = np.asarray(testDF['text'])

test_y = (np.asarray(testDF["sentiment"]).reshape(498,1) /4).astype(int)
train_y = (np.asarray(trainDF["sentiment"]) /4).astype(int)

tweet_w2v = gensim.models.Word2Vec.load(ROOT_DIR + "/Models/word2vec.model")

with open(ROOT_DIR + "/Models/TFIDF.pickle", 'rb') as handle:
    tfidf = load(handle)

embeddings_index = {}

for w in tweet_w2v.wv.vocab.keys():
    embeddings_index[w] = (tweet_w2v.wv[w])
    
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(train_x)
sequences = tokenizer.texts_to_sequences(train_x)

x_train_seq = pad_sequences(sequences, maxlen=280)

test_sequences = tokenizer.texts_to_sequences(test_x)
x_test_seq = pad_sequences(test_sequences, maxlen=250)

num_words = 100000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Input, Dropout, Embedding, MaxPooling1D, LSTM
from keras.callbacks import TensorBoard


print("here")
model_cnn = Sequential()
e = Embedding(100000, 200, weights=[embedding_matrix], input_length=250, trainable=True)
model_cnn.add(e)
model_cnn.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn.add(MaxPooling1D(3))
model_cnn.add(Conv1D(filters=100, kernel_size=3, padding='valid', activation='relu', strides=1))
model_cnn.add(MaxPooling1D(3))
model_cnn.add(Conv1D(filters=100, kernel_size=4, padding='valid', activation='relu', strides=1))
model_cnn.add(Flatten())
model_cnn.add(Dense(256, activation='relu'))
model_cnn.add(Dense(1, activation='sigmoid'))
model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn.fit(x_train_seq, train_y, validation_data=(x_test_seq, test_y), epochs=1, batch_size=256, verbose=1)

model_lstm = Sequential()
model_lstm.add(LSTMz(128, input_shape=(200, 1)))
model_lstm.add(Dense(32, activation='relu'))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.fit(x_train_seq, train_y, validation_data=(x_test_seq, test_y), epochs=1, batch_size=256, verbose=1)


        

        
