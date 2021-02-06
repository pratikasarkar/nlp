# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 00:13:31 2021

@author: Pratik Asarkar
"""

import pandas as pd

df = pd.read_csv("E://NLP//stock-news-data//Data.csv",encoding="ISO-8859-1")

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

data = train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex = True, inplace = True)

list1 = [i for i in range(25)]
new_index = [str(i) for i in list1]
data.columns = new_index

for index in new_index:
    data[index] = data[index].str.lower()
    
headlines = []
for row in range(len(data.index)):
    headlines.append(" ".join(str(x) for x in data.iloc[row,:]))
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

countVectorizer = CountVectorizer(ngram_range=(2,2))
traindataset = countVectorizer.fit_transform(headlines)

randomforestClassifier = RandomForestClassifier(n_estimators=200,criterion="entropy")
randomforestClassifier.fit(traindataset,train['Label'])

test_transform = []
for row in range(len(test.index)):
    test_transform.append(" ".join(str(x) for x in test.iloc[row,:]))
testdataset = countVectorizer.transform(test_transform)

predictions = randomforestClassifier.predict(testdataset)

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
matrix = confusion_matrix(test['Label'],predictions)
accuracy = accuracy_score(test['Label'],predictions)
classification_report = classification_report(test['Label'],predictions)


