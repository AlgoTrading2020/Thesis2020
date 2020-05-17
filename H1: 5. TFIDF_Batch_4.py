#!/usr/bin/env python
# coding: utf-8

#CODE FOR RUNNING TFIDF_CHAR_SAGA (TABLE 16, BATCH 4)

# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def read_data():
    df = pd.read_json('./0.5percent_threshold_20100101-20191231.json')
    return df

df = read_data()

X = df.text
y = df.alpha_label

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

params = {'tfidf__analyzer': ['char'],
          'tfidf__ngram_range': [(2,7), (3,8), (4,9)],
          'tfidf__max_df': [1.0, 0.8],
          'tfidf__min_df': [1, 50],
          'clf__C':[0.001, 1, 4],
          'clf__solver': ['saga'],
          'clf__multi_class':['multinomial'],
          'clf__random_state': [42]
         }
#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('clf', LogisticRegression()),
                ])

grid = GridSearchCV(pipe, params, cv=5, n_jobs=12)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.cv_results_)

results = pd.DataFrame(grid.cv_results_)
results.to_json('./outfile_05data.json')

