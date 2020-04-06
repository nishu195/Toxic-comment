#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# In[2]:


train = pd.read_csv('/home/nishu/nlp project/Toxic comment classifier/train.csv')
test = pd.read_csv('/home/nishu/nlp project/Toxic comment classifier/test.csv')
subm = pd.read_csv('/home/nishu/nlp project/Toxic comment classifier/sample_submission.csv')


# In[3]:


lens = train.comment_text.str.len()


# In[4]:


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)


# In[5]:


COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)


# In[6]:


import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).sp`lit()


# In[7]:


n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
# test_term_doc = vec.transform(test[COMMENT])


# # to save the fit vectorizer ... will be used in prediction...

# In[8]:


import pickle

pickle.dump(vec, open("vec.pickle", "wb"))


# In[9]:


def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)


# In[10]:


x = trn_term_doc
# test_x = test_term_doc


# In[11]:


def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r


# In[12]:


preds = np.zeros((len(test), len(label_cols)))
M = []
for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    M.append([m,r])

#     preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


# # save the list of models for each label in pickle file

# In[14]:


import pickle

pickle.dump(M, open("M.pickle", "wb"))







