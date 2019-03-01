import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

tdata = pd.read_csv('train.tsv',sep='\t')
targets = tdata['Sentiment'].copy()
tdata = tdata['Phrase'].copy()
train = pd.read_csv('test.tsv',sep='\t')


cv = CountVectorizer(binary=True)
X = cv.fit_transform(tdata)
test = train['Phrase'].copy()
X_test = cv.transform(test)

lr = LogisticRegression(C=0.5)
lr.fit(X,targets)
pr = lr.predict(X_test)
pred = pd.DataFrame({'PhraseId' :train['PhraseId'].copy(), 'Sentiment' :pr.astype(int)})
print(pred.head())
pred.to_csv('pred.csv',sep=",",index=False)
