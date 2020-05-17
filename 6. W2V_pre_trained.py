#!/usr/bin/env python
# coding: utf-8

#TABLE 17, WORD2VEC

# In[1]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

import gensim
import numpy as np
import nltk

def read_data():
    df = pd.read_json('./0.5percent_threshold_20100101-20191231.json')
    return df

def lowercase(text):
    #lowercase all words
    text = text.lower()
    return text

def swe():
    #pre-trained model imported from this site: http://vectors.nlpl.eu/
    import zipfile
    repository = "C:/Users/Fredrik/Documents"
    with zipfile.ZipFile(repository + "/69.zip", "r") as archive:
        stream = archive.open("model.txt")
    wv = gensim.models.KeyedVectors.load_word2vec_format(stream, binary=False, unicode_errors='replace')
    
    #replace the raw vectors with the cosine similarity vectors
    wv.init_sims(replace=True)
    return wv

def word_averaging(wv, words):
    #wv is the model, words is the words in a document
    
    #Create a list
    mean = []
    #Iterate over the words
    for word in words:
        
        #Checks if the word is of type np.ndarray (i.e. if each word is already represented by its vectors).
        #If True, append the already extracted vectors to the list "mean".
        #Elif False, first extract and then append the word vectors to the list "mean" for a word 
        #if it is in the word embeddings vocabulary. 
        if isinstance(word, np.ndarray):
            mean.append(word)
        elif word in wv.vocab:
            mean.append(wv.vectors_norm[wv.vocab[word].index])
    
    #if a document contains words that are not included in the word embeddings vocabulary, it won't receive any vectors.
    #In place, return the same number of zeros as there are word dimensions. There are two such rows, 
    #one in train and one in test. Since they are so few, and since they have not been removed for other models, 
    #we do not remove them during the model evaluation step. 
    if not mean:
        return np.zeros(wv.vector_size,)
    
    #Each document has varying number of words, meaning in each "mean" list there are 
    #(no. of words) * (no. of dimensions for each word), e.g. 1683 * 100 in one document and 843 * 100 in another.
    #The below code snippet averages each dimension, e.g. creates 100 np.float32 numbers per document.
    mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    return mean

def word_averaging_list(wv, corpus):
    
    #[word_averaging(wv, document) for document in corpus] returns a list (corpus) of arrays (documents), 
    #Each document ("mean") is one array
    #np.vstack concatenates these arrays, instead returning an array (corpus) of lists (documents).
    return np.vstack([word_averaging(wv, document) for document in corpus])

def w2v_tokenize_text(text):
    tokens = []
    
    #Using the nltk package for a more advanced tokenizing of (1) sentences and (2) words
    #as they are usually written in Swedish, to possibly improve quality compared to using mere spaces.
    for sent in nltk.sent_tokenize(text, language='swedish'):
        for word in nltk.word_tokenize(sent, language='swedish'):
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens

def transform(text, wv):
#Create a numpy array (corpus) of lists (documents) of tokens (words)
#Subsequently average each dimension (100) of all words in each document
    tokenized = text.apply(lambda document: w2v_tokenize_text(document)).values
    word_average = word_averaging_list(wv, tokenized)
    return word_average
    
#import dataset
df = read_data()

X = df.text
y = df.alpha_label
X = X.apply(lowercase)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2, stratify=y)

#import pre-trained model and vocabulary
wv = swe()

X_train_transformed = transform(X_train, wv)

params={'clf__solver': ['liblinear', 'saga'],
        'clf__multi_class':['auto'],
        'clf__C':[0.001, 1, 4],
        'clf__random_state':[42]}

pipe = Pipeline([('clf', LogisticRegression())
                ])

grid = GridSearchCV(pipe, params, cv=5, n_jobs=-1)
grid.fit(X_train_transformed, y_train)

results = pd.DataFrame(grid.cv_results_)
results

