#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
pd.set_option('max_columns',200)


# In[2]:


df=pd.read_csv('coaster_db (1).csv')
df.head()


# In[3]:


df.dtypes


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


df.columns


# In[8]:


df.describe()


# In[9]:


df.corr()


# In[10]:


plt.figure(figsize=(12,8))
sns.heatmap(data=df.corr(),annot=True)
plt.show()


# In[11]:


df.columns 


# In[12]:


data=df[['coaster_name','Location', 'Status', 'Manufacturer', 'year_introduced','latitude', 'longitude', 'Type_Main',
       'opening_date_clean','speed_mph','height_ft','Inversions_clean', 'Gforce_clean']]


# In[13]:


data.head()


# In[14]:


data.shape


# In[15]:


data.dtypes


# In[16]:


(data['opening_date_clean'])=pd.to_datetime(data['opening_date_clean'])


# In[17]:


#Rename columns
data.rename(columns=
           {'coaster_name ':'Coaster_Name'})


# In[18]:


data.isnull().sum()


# In[19]:


data.loc[data.duplicated(subset='coaster_name')]


# In[20]:


data['coaster_name']


# In[21]:


data.columns


# In[22]:


data=data.loc[~data.duplicated(subset=['coaster_name', 'Location','opening_date_clean'])].reset_index(drop=True).copy()


# In[23]:


data.shape


# In[24]:


data.isnull().sum()


# In[25]:


ax=data['year_introduced'].value_counts().head(10).plot(kind='bar',title='Top Year Coasters')
ax.set_xlabel('year_introduced')
ax.set_ylabel('count')


# In[26]:


data['speed_mph'].plot(kind='hist',bins=20,title='Speed 0f Coaster')


# # Feature Relationship

# In[27]:


data.plot(kind='scatter',x='speed_mph',y='height_ft',title='Coaster Speed vs Height')
plt.show()


# In[28]:


sns.scatterplot(x='speed_mph',y='height_ft',data=data,hue='year_introduced')


# In[29]:


data.columns


# In[30]:


#To comapre multiple dataset

sns.pairplot(data,vars=['latitude', 'longitude',  'year_introduced', 'speed_mph', 'height_ft', 'Inversions_clean', 'Gforce_clean'])


# In[31]:


data_corr = data[['coaster_name', 'Location', 'Status', 'Manufacturer', 'year_introduced',
       'latitude', 'longitude', 'Type_Main', 'opening_date_clean', 'speed_mph',
       'height_ft', 'Inversions_clean', 'Gforce_clean']].dropna().corr()

data_corr


# In[32]:


sns.heatmap(data_corr,annot= True)


# Asking Questions:
#     1.What are the locations with the fastest roller coasters (minimum of 10)?

# In[33]:


ax = df.query('Location != "Other"')     .groupby('Location')['speed_mph']     .agg(['mean','count'])     .query('count >= 10')     .sort_values('mean')['mean']     .plot(kind='barh', figsize=(12, 5), title='Average Coast Speed by Location')
ax.set_xlabel('Average Coaster Speed')
plt.show()


# In[ ]:





# In[ ]:




