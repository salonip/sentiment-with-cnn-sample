#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import gensim
from sklearn.model_selection import train_test_split
import string
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
strlen = lambda x: len(x)


# In[12]:


class TextClassification:
    def __init__(self,vec_len=50):
        self.punctuations = list(string.punctuation)
        self.stop_words = list(set(stopwords.words('english')))
        self.remove_list = self.stop_words + self.punctuations + ["`", "'", "``", "''",","]
        self.nonascii = lambda text: ''.join([i if ord(i) < 128 else ' ' for i in text])
        self.vec_len =vec_len
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format("../res/GoogleNews-vectors-negative300.bin.gz", binary=True)


    def remove_stopwords(self,lines):
        prep = lambda x:' '.join([i for i in str(x).lower().split() if i not in self.remove_list])
        return list(map(prep, lines))

    def remove_spaces(self,lines):
        result=[]
        for line in lines:
            line = ' '.join(line.split(' ')[4:-1])
            result=result+([str(''.join(i for i in line if not i.isdigit()))])
        return result



    def cnvt_word2vec(self,lines,fixed_sent_len):
        vec_len=self.vec_len
        default_value = np.zeros(vec_len)
        if self.stop_words is not None:
            lines = self.remove_stopwords(lines)


        key_set = self.w2v_model.vocab.keys()
        mat = np.zeros((len(lines), fixed_sent_len, vec_len))

        for i in range(0, len(lines)):

            line = lines[i]
            print ("(" + str(round(float(i)/len(lines) , 4)*100) + "%) completed")
            for j in range(0, fixed_sent_len):
                # j = 0
                if j < len(line) and line[j] in key_set:
                    mat[i,j,:] = self.w2v_model[line[j]][0:vec_len]
                else:
                    mat[i,j,:] = default_value
            print ("working with "+str(i) + " of "+ str(len(lines)))
        return mat


# In[13]:




# In[ ]:




