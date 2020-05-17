#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

def read_data():
    df = pd.read_json('C:/Users/Fredrik/Documents/Datasets_uppsats/0.5percent_threshold_20100101-20191231.json')
    return df

df = read_data()


# In[3]:



from sklearn.linear_model import LogisticRegression

X = df.text
y = df.alpha_label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('clf', LogisticRegression(random_state = 42)),
                ])

grid = cross_validate(pipe, , cv=5)
grid.fit(X_train, y_train)

print('Best cross-validation accuracy: %s' % grid.best_score_)

print(y_train.value_counts(' '))

difference = grid.best_score_ - y_train.value_counts(' ')[0]
print('Difference between accuracy and baseline: ', difference)
print(' ')


# In[4]:



from sklearn.naive_bayes import MultinomialNB

X = df.text
y = df.alpha_label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)
params = {}

#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('clf', MultinomialNB()),
                ])

grid = GridSearchCV(pipe, params, cv=5)
grid.fit(X_train, y_train)

print('Best cross-validation accuracy: %s' % grid.best_score_)

print(y_train.value_counts(' '))

difference = grid.best_score_ - y_train.value_counts(' ')[0]
print('Difference between accuracy and baseline: ', difference)
print(' ')


# In[3]:



from sklearn.linear_model import SGDClassifier
params = {'clf__random_state':[42]}

X = df.text
y = df.alpha_label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('clf', SGDClassifier()),
                ])

grid = GridSearchCV(pipe, params, cv=5)
grid.fit(X_train, y_train)

print('Best cross-validation accuracy: %s' % grid.best_score_)

print(y_train.value_counts(' '))

difference = grid.best_score_ - y_train.value_counts(' ')[0]
print('Difference between accuracy and baseline: ', difference)
print(' ')


# In[4]:



from sklearn.svm import LinearSVC
params = {'clf__random_state':[42]}

X = df.text
y = df.alpha_label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('clf', LinearSVC()),
                ])

grid = GridSearchCV(pipe, params, cv=5)
grid.fit(X_train, y_train)

print('Best cross-validation accuracy: %s' % grid.best_score_)

print(y_train.value_counts(' '))

difference = grid.best_score_ - y_train.value_counts(' ')[0]
print('Difference between accuracy and baseline: ', difference)
print(' ')


# In[2]:




from sklearn.svm import SVC
params = {'clf__random_state':[42]}

X = df.text
y = df.alpha_label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('clf', SVC()),
                ])

grid = GridSearchCV(pipe, params, cv=5)
grid.fit(X_train, y_train)

print('Best cross-validation accuracy: %s' % grid.best_score_)

print(y_train.value_counts(' '))

difference = grid.best_score_ - y_train.value_counts(' ')[0]
print('Difference between accuracy and baseline: ', difference)
print(' ')


# In[ ]:




