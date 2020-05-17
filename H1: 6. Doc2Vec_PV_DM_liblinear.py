#!/usr/bin/env python
# coding: utf-8

#TABLE 17, DOC2VEC 4

# In[ ]:


import pandas as pd
from gensim.sklearn_api import D2VTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

def read_data():
    df = pd.read_json('./0.5percent_threshold_20100101-20191231.json')
    return df

def lowercase(text):
    #lowercase all words. TFIDF handles this parameter-wise (default), but D2VTransformer does not.
    text = text.lower()
    return text

df = read_data()

X = df.text
y = df.alpha_label

X = X.apply(lowercase)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

#Transform the dataset from Pandas series to Python list of lists, 
#which is the necessary input for the Doc2Vec fit_transform method
X_train = [document.split() for document in X_train]

params={'d2v__dm':[1],
        'd2v__size':[50, 100, 300],
        'd2v__window':[2, 5, 8],
        'd2v__iter':[1, 5, 15, 30], 
        'd2v__seed':[42], 
        'd2v__workers': [16],
        'clf__solver': ['liblinear'],
        'clf__multi_class':['ovr'],
        'clf__C':[0.001, 1, 4],
        'clf__random_state':[42]}

pipe = Pipeline([('d2v', D2VTransformer()),
                 ('clf', LogisticRegression())
                ])

grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.cv_results_)

results = pd.DataFrame(grid.cv_results_)
results.to_json('./d2v_outfile_05.json')

