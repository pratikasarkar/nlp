# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:42:45 2021

@author: ASUS
"""

import pandas as pd
df = pd.read_csv(r'D:\nlp\fake-news-data\train.csv')

df = df.dropna()

X = df.drop('label',axis = 1)
y = df['label']

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

# Vocabulary size
voc_size = 5000

# One Hot Representation
messages = X.copy()
messages.reset_index(inplace = True)

import nltk
import re
from nltk.corpus import stopwords

# Dataset Preprocessing
from nltk.stem import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    review = " ".join(review)
    corpus.append(review)
    
onehot_repr = [one_hot(words,voc_size) for words in corpus]
    
sent_len = 20
embedded_doc = pad_sequences(onehot_repr,maxlen = sent_len,padding = 'pre')

# Creating the model
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_len))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.summary()

import numpy as np
X_final = np.array(embedded_doc)
y_final = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final,y_final,test_size = 0.33,random_state = 42)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

y_pred = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
