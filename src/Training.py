#!/usr/bin/env python
# coding: utf-8

# In[30]:


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


# In[66]:


dic = dict({0:'negative', 1:'positive'})


# In[32]:


tc = dp.TextClassification(100)


# In[47]:


def create_sets(data_file) :
    data = pd.read_csv(data_file,sep=',',encoding='ISO-8859-1')
    data['clean'] = tc.remove_stopwords(data['text'])
    #data['clean'] = tc.remove_spaces(data['clean'])
    #data = data.drop_duplicates(subset=['clean'], keep='first')
    x_train = data['clean']
    y_train = data['category']
    x_train, x_test, y_train, y_test = train_test_split(list(x_train), list(y_train), test_size=0.10, random_state=42)


    df_test = pd.DataFrame({'Category':y_test,'Text':x_test})
    df_test.to_csv('../out/test.tsv',sep='\t')

    df = pd.DataFrame({'Category':y_train,'Text':x_train})
    df.to_csv('../out/train.txt',sep='\t')

    train_labels= df['Category']
    test_labels= df_test['Category']
    x_train = list(df['Text'])
    x_test = list(df_test['Text'])
    y_train= list(train_labels)
    y_test= list(test_labels)
    return x_train,y_train,x_test,y_test


# In[48]:


data_file = "../res/yelp_labelled.csv" 
x_train,y_train,x_test,y_test = create_sets(data_file)


# In[49]:


def fix_len(lines,stop_words):
    if stop_words is not None:
        lines = tc.remove_stopwords(lines)
    return list(map(dp.strlen,lines))
    


# In[50]:


len_sents  = fix_len(x_train,tc.remove_list)
plt.plot(range(0,len(len_sents)),len_sents)
plt.show()
x= pd.DataFrame(len_sents)
m = np.asarray(len_sents).mean()
s = np.asarray(len_sents).std()
fixed_sent_len =int((m-s)/2)
#fixed_sent_len = int(x.describe().iloc[-2][0])
#fixed_sent_len = int(m)
fixed_sent_len


# In[51]:


matrix = tc.cnvt_word2vec(x_train,fixed_sent_len =fixed_sent_len)
np.save("../out/x_train.npy", matrix)

matrix = tc.cnvt_word2vec(x_test,fixed_sent_len =fixed_sent_len)
np.save("../out/x_test.npy", matrix)

np.save("../out/y_test.npy", np.asarray(y_test))
np.save("../out/y_train.npy", np.asarray(y_train))


# In[52]:


np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Flatten

from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import keras
from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph())
K.set_session(sess)
num_epoch = 100
num_classes = 2
batch_size = 10
rows, cols = tc.vec_len, fixed_sent_len
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# In[53]:


def load_data(basepath):
    return np.load(basepath+"/x_train.npy"),            np.load(basepath+"/x_test.npy"),            np.load(basepath+"/y_train.npy"),            np.load(basepath+"/y_test.npy")

def create_convnet(img_path='../out/cnn_model_word2vec.png'):
    inputs = Input(shape=(rows, cols, 1))

    tower_1 = Conv2D(100, (rows, 3), padding='valid', activation='relu')(inputs)
    tower_1 = MaxPooling2D((1, cols-3+1), strides=(1, 1), padding='valid')(tower_1)

    tower_2 = Conv2D(100, (rows, 4), padding='valid', activation='relu')(inputs)
    tower_2 = MaxPooling2D((1, cols-4+1), strides=(1, 1), padding='valid')(tower_2)

    
    tower_3 = Conv2D(100, (rows, 5), padding='valid', activation='relu')(inputs)
    tower_3 = MaxPooling2D((1, cols-5+1), strides=(1, 1), padding='valid')(tower_3)

    merged = keras.layers.concatenate([tower_1, tower_2,tower_3], axis=1)
    merged = Flatten()(merged)
    
   # dense_1 = Dense(100, activation='tanh')(merged)
    out = Dense(num_classes, activation='softmax')(merged)
    

    model = Model(inputs, out)
    #from keras.utils.vis_utils import plot_model
    #plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)
    return model


# In[54]:


x_train, x_test, y_train, y_test = load_data(basepath)
x_test.shape
x_train = x_train.reshape(x_train.shape[0], rows, cols,1)
x_test = x_test.reshape(x_test.shape[0], rows, cols,1)

#y_train = [dic[x] for x in y_train]

#y_test = [dic[x] for x in y_test]

y_train = np_utils.to_categorical(y_train)

y_test = np_utils.to_categorical(y_test)


# In[55]:


some_model = create_convnet()


# In[56]:


from keras.optimizers import SGD,Adam
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.98, nesterov=True)
adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# In[57]:


some_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=adam,metrics=['accuracy'])


# In[58]:


filepath="../out/cnn_model_word2vec"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[59]:


history = some_model.fit(x_train, y_train,
              epochs=num_epoch,
              verbose=1,
              validation_data=(x_test, y_test),
              #validation_split=0.20,
              callbacks=callbacks_list)


# In[62]:


fig2=plt.figure()
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves : CNN',fontsize=16)
fig2.savefig('accuracy_cnn.png')
plt.show()


# In[20]:



def train(basepath):
    x_train, x_test, y_train, y_test = load_data(basepath)
    x_test.shape

    x_test = x_test.reshape(x_test.shape[0], rows, cols,1)

    y_train = [dic[x] for x in y_train]

    y_test = [dic[x] for x in y_test]

    y_train = np_utils.to_categorical(y_train)

    y_test = np_utils.to_categorical(y_test)
    some_model = create_convnet()    
    some_model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=adam,
                  metrics=['accuracy'])

    filepath="../out/cnn_model_word2vec"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    import gc
    gc.collect()
    
    history = some_model.fit(x_train, y_train,
              epochs=num_epoch,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=callbacks_list)
    fig1 = plt.figure()
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves :CNN',fontsize=16)
    fig1.savefig('../out/loss_cnn.png')
    plt.show()
    K.clear_session()
    


# In[63]:


def test(filepath,basepath):
    x_train, x_test, y_train, y_test = load_data(basepath)
    model_load = keras.models.load_model(filepath)
    x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)
    y_test = np_utils.to_categorical(y_test)
    score = model_load.evaluate(x_test, y_test, verbose=0)
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    model_score = model_load.predict(x_test)
    np.savetxt("../out/predicted_scores",model_score)
    
# =============================================================================
    
    from sklearn.metrics import confusion_matrix
    model_preds = np.argmax(model_score, axis=1)
    actual_scores= np.argmax(y_test, axis=1)
    print(confusion_matrix(actual_scores, model_preds))   
    actual_scores = [dic[x] for x in actual_scores]
    import pandas as pd
    df = pd.DataFrame({'Actual':actual_scores,'Predicted':model_preds})
    df.to_csv('../out/Pred_results.csv',sep='\t')
# =============================================================================


# In[64]:


basepath = "../out/"


# In[67]:


#train(basepath)
filepath="../out/cnn_model_word2vec"
test(filepath,basepath)


# In[ ]:





# In[ ]:




