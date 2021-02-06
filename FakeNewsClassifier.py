# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 08:51:16 2021

@author: Pratik Asarkar
"""

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier

df = pd.read_csv("E://NLP//fake-news-data//train.csv")

X = df.drop('label',axis = 1)

df = df.dropna()

messages = df.copy()

messages.reset_index(inplace=True)

ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)    
    
    
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()

y = messages['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)

def plot_confusion_matrix(cm,classes,normalize=False,title = "Confusion Matrix",cmap = plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation = 45)
    plt.yticks(tick_marks,classes)
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion Matrix, without normalization")
        
    thresh = cm.max()/2.
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment='center',color = 'white' if cm[i,j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel("Predicted Label")
    
classifier = MultinomialNB()
classifier.fit(X_train,y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
cm = metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm,classes=['FAKE','REAL']) 

pa_classifier = PassiveAggressiveClassifier(n_iter_no_change = 50)
pa_classifier.fit(X_train,y_train)
pred = pa_classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
cm = metrics.confusion_matrix(y_test,pred)
plot_confusion_matrix(cm,classes=['FAKE','REAL'])

classifier = MultinomialNB(alpha = 0.1)
prev_score = 0
for alpha in np.arange(0,1,0.1):
    sub_classifier = MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred = sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>prev_score:
        classifier = sub_classifier
    print("Alpha : {}, Score : {}".format(alpha,score))
    
feature_names = cv.get_feature_names()
print(sorted(zip(classifier.coef_[0],feature_names))[:20]) # most fake words
