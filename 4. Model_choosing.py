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


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import LogisticRegression\n\nX = df.text\ny = df.alpha_label\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)\n\n#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.\npipe = Pipeline([('tfidf', TfidfVectorizer()),\n                 ('clf', LogisticRegression(random_state = 42)),\n                ])\n\ngrid = cross_validate(pipe, , cv=5)\ngrid.fit(X_train, y_train)\n\nprint('Best cross-validation accuracy: %s' % grid.best_score_)\n\nprint(y_train.value_counts(' '))\n\ndifference = grid.best_score_ - y_train.value_counts(' ')[0]\nprint('Difference between accuracy and baseline: ', difference)\nprint(' ')")


# In[4]:


get_ipython().run_cell_magic('time', '', "from sklearn.naive_bayes import MultinomialNB\n\nX = df.text\ny = df.alpha_label\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)\nparams = {}\n\n#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.\npipe = Pipeline([('tfidf', TfidfVectorizer()),\n                 ('clf', MultinomialNB()),\n                ])\n\ngrid = GridSearchCV(pipe, params, cv=5)\ngrid.fit(X_train, y_train)\n\nprint('Best cross-validation accuracy: %s' % grid.best_score_)\n\nprint(y_train.value_counts(' '))\n\ndifference = grid.best_score_ - y_train.value_counts(' ')[0]\nprint('Difference between accuracy and baseline: ', difference)\nprint(' ')")


# In[3]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import SGDClassifier\nparams = {'clf__random_state':[42]}\n\nX = df.text\ny = df.alpha_label\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)\n\n#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.\npipe = Pipeline([('tfidf', TfidfVectorizer()),\n                 ('clf', SGDClassifier()),\n                ])\n\ngrid = GridSearchCV(pipe, params, cv=5)\ngrid.fit(X_train, y_train)\n\nprint('Best cross-validation accuracy: %s' % grid.best_score_)\n\nprint(y_train.value_counts(' '))\n\ndifference = grid.best_score_ - y_train.value_counts(' ')[0]\nprint('Difference between accuracy and baseline: ', difference)\nprint(' ')")


# In[4]:


get_ipython().run_cell_magic('time', '', "from sklearn.svm import LinearSVC\nparams = {'clf__random_state':[42]}\n\nX = df.text\ny = df.alpha_label\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)\n\n#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.\npipe = Pipeline([('tfidf', TfidfVectorizer()),\n                 ('clf', LinearSVC()),\n                ])\n\ngrid = GridSearchCV(pipe, params, cv=5)\ngrid.fit(X_train, y_train)\n\nprint('Best cross-validation accuracy: %s' % grid.best_score_)\n\nprint(y_train.value_counts(' '))\n\ndifference = grid.best_score_ - y_train.value_counts(' ')[0]\nprint('Difference between accuracy and baseline: ', difference)\nprint(' ')")


# In[2]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.svm import SVC\nparams = {'clf__random_state':[42]}\n\nX = df.text\ny = df.alpha_label\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)\n\n#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.\npipe = Pipeline([('tfidf', TfidfVectorizer()),\n                 ('clf', SVC()),\n                ])\n\ngrid = GridSearchCV(pipe, params, cv=5)\ngrid.fit(X_train, y_train)\n\nprint('Best cross-validation accuracy: %s' % grid.best_score_)\n\nprint(y_train.value_counts(' '))\n\ndifference = grid.best_score_ - y_train.value_counts(' ')[0]\nprint('Difference between accuracy and baseline: ', difference)\nprint(' ')")


# In[ ]:




