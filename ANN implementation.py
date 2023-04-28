#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing  import StandardScaler 
from sklearn.model_selection import train_test_split

import seaborn as sns


# In[4]:


from keras.models import Sequential
from keras.layers import Dense,BatchNormalization,Dropout,LSTM
from keras import callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score,recall_score,confusion_matrix, classification_report, accuracy_score, f1_score


# In[7]:


df= pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.head()


# In[10]:


df.info()


# In[11]:


sns.countplot(x=df['DEATH_EVENT'])


# In[13]:


X=df.drop(['DEATH_EVENT'],axis=1)
y=df['DEATH_EVENT']


# In[16]:


cols_name=list(X.columns)
s_scaler= preprocessing.StandardScaler()
X_df=s_scaler.fit_transform(X)
X_df=pd.DataFrame(X_df,columns=cols_name)
X_df.describe().T


# In[17]:


X_train,X_test,y_train,y_test= train_test_split(X_df,y,test_size=0.25,random_state=7)


# In[18]:


ear_stop=callbacks.EarlyStopping(
        min_delta=0.001,patience=30,restore_best_weights=True)

model = Sequential()

model.add(Dense(units =16,kernel_initializer ='uniform',activation='relu',input_dim=12))
model.add(Dense(units=8,kernel_initializer='uniform',activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

from tensorflow.keras.optimizers import SGD

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[21]:


history = model.fit(X_train,y_train,batch_size=32,epochs=500,callbacks=[ear_stop], validation_split=0.2)


# In[22]:


val_accuracy=np.mean(history.history['val_accuracy'])
print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy*100))


# In[23]:


y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)
np.set_printoptions()


# In[24]:


y_test


# In[26]:


plt.subplots(figsize=(12,8))
cf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix),annot=True,annot_kws = {'size':20})


# In[ ]:




