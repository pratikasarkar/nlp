# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:59:45 2021

@author: ASUS
"""

# Word Embedding Techniques using Embedding Layer in Keras

from tensorflow.keras.preprocessing.text import one_hot

sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',
     'a king',
     'a queen']

# Vocabulary size
voc_size = 10000

# One hot representation
onehot_repr = [one_hot(words,voc_size) for words in sent]

# Word Embedding Representation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

sent_len = 8
embedded_docs = pad_sequences(onehot_repr,padding = 'pre',maxlen = sent_len)

dim = 10

model = Sequential()
model.add(Embedding(voc_size,dim,input_length=sent_len))
model.compile(optimizer = 'adam', loss = 'mse')

model.summary()

model.predict(embedded_docs)[[7,8]]
