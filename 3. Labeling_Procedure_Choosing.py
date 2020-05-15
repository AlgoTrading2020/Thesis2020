#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

def read_data():
    df05 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\0.5percent_threshold_20100101-20191231.json')
    df10 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\1.0percent_threshold_20100101-20191231.json')
    df15 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\1.5percent_threshold_20100101-20191231.json')
    df20 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\2.0percent_threshold_20100101-20191231.json')
    df25 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\2.5percent_threshold_20100101-20191231.json')
    df30 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\3.0percent_threshold_20100101-20191231.json')
    df35 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\3.5percent_threshold_20100101-20191231.json')
    df40 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\4.0percent_threshold_20100101-20191231.json')
    df45 = pd.read_json(r'C:\Users\Fredrik\Documents\Datasets_uppsats\4.5percent_threshold_20100101-20191231.json')
    return df05,df10,df15,df20,df25,df30,df35,df40,df45

datasets = list(read_data())


# In[2]:


def PM(datasets):
    count=0
    for dataset in datasets:
        count+=5
        X = dataset.text
        y = dataset.PM_label
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 0)

        params = {'clf__random_state':[0]}

        #TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
        pipe = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LogisticRegression()),
                        ])
        
        grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        print('df'+str(count))
        print('Best cross-validation accuracy: %s' % grid.best_score_)
        
        print(y_train.value_counts(' '))
        
        difference = grid.best_score_ - y_train.value_counts(' ')[0]
        print('Difference between accuracy and baseline: ', difference)
        print(' ')

PM(datasets)


# In[4]:


def PM_index_adjusted_label(datasets):
    count=0
    for dataset in datasets:
        count+=5
        X = dataset.text
        y = dataset.PM_index_adjusted_label

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 0)

        params = {'clf__random_state':[0]}

        #TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
        pipe = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LogisticRegression()),
                        ])
        
        grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        print('df'+str(count))
        print('Best cross-validation accuracy: %s' % grid.best_score_)
        
        print(y_train.value_counts(' '))
        
        difference = grid.best_score_ - y_train.value_counts(' ')[0]
        print('Difference between accuracy and baseline: ', difference)
        print(' ')
        
PM_index_adjusted_label(datasets)


# In[3]:


def alpha_label(datasets):
    count=0
    for dataset in datasets:
        count+=5
        X = dataset.text
        y = dataset.alpha_label

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 0)

        params = {'clf__random_state':[0]}

        #TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
        pipe = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LogisticRegression()),
                        ])
        
        grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        print('df'+str(count))
        print('Best cross-validation accuracy: %s' % grid.best_score_)
        
        print(y_train.value_counts(' '))
        
        difference = grid.best_score_ - y_train.value_counts(' ')[0]
        print('Difference between accuracy and baseline: ', difference)
        print(' ')
        
alpha_label(datasets)


# In[ ]:




