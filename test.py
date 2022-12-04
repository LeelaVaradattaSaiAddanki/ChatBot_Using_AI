from nltk.stem import WordNetLemmatizer
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import random
import json
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

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

model = tf.keras.models.load_model("ChatBot.h5")

def input_bag(sen, words):
    bag = [0]*len(words)
    wrds = nltk.word_tokenize(sen)
    wrds = [lemm.lemmatize(w.lower()) for w in wrds]
    for w in wrds:
        for i, j in enumerate(words):
            if j == w:
                bag[i] = 1
    return np.array(bag)


def response():
    print('MONA: Hey there! I am MONA, the chatbot that can answer all your queries. Type exit to leave.')
    while True:
        inputs = input('You:')
        if inputs.lower() == 'exit':
            break
        x = input_bag(inputs, words)
        res = model.predict(np.array([x]))[0]
        pred_index = np.argmax(res)
        tag = labels[pred_index]
        for t in data['intents']:
            if t['intent'] == tag:
                resp = t['responses']
                break
        print("MONA: "+random.choice(resp))
