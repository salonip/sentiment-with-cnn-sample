#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import gensim
from sklearn.model_selection import train_test_split
import string
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
import Processing as dp
fixed_sent_len=7
filepath="../out/cnn_model_word2vec"
import keras


# In[14]:


tc = dp.TextClassification(100)


# In[15]:


article='Manipal Hospitals-Medanta deal will take more time to close, says top executive'


# In[21]:


rows, cols = tc.vec_len, fixed_sent_len
dic = dict({0:'negative', 1:'positive'})

article = tc.remove_stopwords(article)
matrix = tc.cnvt_word2vec([article],fixed_sent_len =fixed_sent_len)
matrix = matrix.reshape(matrix.shape[0], rows, cols,1)

model= keras.models.load_model(filepath)
predictions =  model.predict(matrix)
actual_scores= np.argmax(predictions, axis=1)
actual_scores = [dic[x] for x in actual_scores]


# In[22]:


actual_scores


# In[ ]:




