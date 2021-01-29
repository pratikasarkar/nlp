# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:44:50 2021

@author: Pratik Asarkar
"""

import pandas as pd
messages = pd.read_csv('SMSSpamCollection',sep='\t', names=['label','message'])

import nltk               
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus = []
for i in range(len(messages)):
    review = re.sub("[^a-zA-Z]"," ",messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    #review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)
    
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 5000)
#X = cv.fit_transform(corpus).toarray()
    
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(max_features=5000)
X = tfidfVectorizer.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detection_model = MultinomialNB().fit(X_train,y_train)

y_pred = spam_detection_model.predict(X_test) 

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred) 
confusion_matrix = metrics.confusion_matrix(y_test,y_pred)