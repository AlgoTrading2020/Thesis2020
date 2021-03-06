#!/usr/bin/env python
# coding: utf-8

#CODE FOR RUNNING TFIDF_CHAR_LIBLINEAR (TABLE 16, BATCH 3)

# In[3]:


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
          'tfidf__ngram_range': [(4,9), (3,8), (2,7)],
          'tfidf__max_df': [1.0, 0.8],
          'tfidf__min_df': [1, 50],
          'clf__C':[0.001, 1, 4],
          'clf__solver': ['liblinear'],
          'clf__multi_class':['ovr'],
          'clf__random_state': [42]
         }
#TfidfVectorizer (1) vectorizes and computes TF, (2) computes IDF and (3) computes TFIDF for each word.
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('clf', LogisticRegression()),
                ])

grid = GridSearchCV(pipe, params, cv=5, n_jobs=8)
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.cv_results_)

results = pd.DataFrame(grid.cv_results_)
results.to_json('./outfile_05data.json')


# In[ ]:




