#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from joblib import dump

def read_data():
    df = pd.read_json('C:/Users/Fredrik/Documents/Datasets_uppsats/0.5percent_threshold_20100101-20191231.json')
    return df

df = read_data()

X = df.text
y = df.alpha_label


# In[ ]:


#Evaluate best model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

tfidf = TfidfVectorizer(analyzer='char', ngram_range = (4,9), max_df = 0.8, min_df = 1)
X_train_transformed = tfidf.fit_transform(X_train)
X_test_transformed = tfidf.transform(X_test)

logreg = LogisticRegression(C=1, solver='saga', multi_class = 'multinomial')
logreg = logreg.fit(X_train_transformed, y_train)

y_pred = logreg.predict(X_test_transformed)
print(confusion_matrix(y_true = y_test, y_pred = y_pred))
print(classification_report(y_true = y_test, y_pred = y_pred))
print(accuracy_score(y_test, y_pred))

dump(tfidf, './tfidf_05dataset.joblib')
dump(logreg, './logreg_05dataset.joblib')

