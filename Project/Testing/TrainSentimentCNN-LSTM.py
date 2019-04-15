import numpy as np
import pandas as pd
from numpy import asarray
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Input, Dropout, LSTM
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading in data....")
cols = ['sentiment','text']
trainDF = pd.read_csv(ROOT_DIR + "/Data/CleanedData/trainingdata.csv",header=None, names=cols, encoding='latin-1')
trainDF = trainDF.iloc[1:]
trainDF = trainDF.dropna()

testDF = pd.read_csv(ROOT_DIR + "/Data/CleanedData/testingdata.csv",header=None, names=cols, encoding='latin-1')
testDF = testDF.iloc[1:]
testDF = testDF.dropna()

test_y = asarray(testDF["sentiment"]).reshape(498,1) /4
train_y = asarray(trainDF["sentiment"]) /4
del testDF, trainDF

train_x = np.load(ROOT_DIR + "/Data/VectorsData/trainingTweetVectors.npy")
test_x = np.load(ROOT_DIR + "/Data/VectorsData/testingTweetVectors.npy")

delete = list(~np.all(test_y == 0.5, axis=1))
test_y = test_y[delete]
test_x = test_x[delete]

train_x = np.expand_dims(train_x, axis=2)
test_x = np.expand_dims(test_x, axis=2)

model = Sequential()
model.add(Dropout(0.25,    
                  input_shape=(100, 1)))
model.add(Conv1D(filters=50, kernel_size=4 ,      
                  activation= 'relu'))
model.add(MaxPooling1D(3))
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_x, train_y, epochs=5, batch_size=64, verbose=1)


predictions = model.predict(test_x, batch_size=64, verbose=1)

predictions = np.round(predictions)

acc = sum(1 for x,y in zip(predictions,test_y) if x == y) / len(predictions)

# serialize model to JSON
model_json = model.to_json()
with open("Models/SentimentNN.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Models/SentimentNN.h5")
print("Saved model to disk")

