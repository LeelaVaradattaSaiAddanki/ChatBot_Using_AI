import json
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer 
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

with open('/content/Intents.json') as file:
    data=json.load(file,strict=False)

lemm=WordNetLemmatizer()
words = []
labels = []
X = []
Y = []
for intent in data['intents']:
  for text in intent['text']:
    w = nltk.word_tokenize(text)
    words.append(w)
    X.append((w, intent['intent']))

    if intent['intent'] not in labels:
      labels.append(intent['intent'])

words = [lemm.lemmatize(j.lower()) for i in words for j in i if j not in [' ', '.', '?']]
words=sorted(list(set(words)))
labels=sorted(list(set(labels)))
train=[]
output=[0]*len(labels)
for doc in X:
    bag=[]
    text = doc[0]
    text = [lemm.lemmatize(w.lower()) for w in text]
    for w in words:
      if w in text:
        bag.append(1)
      else:
        bag.append(0)
    output_row=list(output)
    output_row[labels.index(doc[1])] = 1
    train.append((bag,output_row))

random.shuffle(train)
train=np.array(train)
train_x=list(train[:,0])
train_y=list(train[:,1])
train=np.array(train)
output=np.array(output)
train_x=np.asarray(train_x)
train_y=np.asarray(train_y)


# Model
model = Sequential()

model.add(Dense(64, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(rate = 0.35))

model.add(Dense(64, activation='relu'))
model.add(Dropout(rate = 0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(rate = 0.5))

model.add(Dense(len(train_y[0]), activation="softmax"))
model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

model.fit(train_x,train_y,epochs=150,verbose=1,batch_size=3)
model.save('ChatBot.h5')