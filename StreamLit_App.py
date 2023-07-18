#!/usr/bin/env python
# coding: utf-8

# # Data: 
# 
# ***Wine Quality Dataset: https://www.kaggle.com/datasets/subhajournal/wine-quality-data-combined***

# In[1]:


import pandas as pd

df = pd.read_csv('WineQuality.csv')
df


# # Dropping column titled 'Unnamed: 0'

# In[2]:


df = df.drop('Unnamed: 0', axis = 1)
df


# # Basic Properties of the Data
# 
# ## 1. Data Types

# In[3]:


df.dtypes


# ***Type is non-numeric categorical, Quality is numeric categorical***
# 
# ## 2. Unique Types of Wine

# In[4]:


df['Type'].unique()


# In[5]:


df['quality'].unique()


# # Encoding Wine Types & Quality

# In[6]:


df.loc[df["Type"] == "White Wine", "Type"] = 1
df.loc[df["Type"] == "Red Wine", "Type"] = 0


# In[7]:


df


# # Prediction: Linear Discriminant Analysis

# In[8]:


from sklearn.model_selection import train_test_split

X = df.drop('quality', axis = 1)
y = df['quality']
X2 = X.copy()


# In[9]:


y


# In[10]:


# Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# In[11]:


X_train.shape


# In[12]:


X_test.shape


# In[13]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[14]:


lda = LinearDiscriminantAnalysis(n_components=1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# In[15]:


# classify using random forest classifier
classifier = RandomForestClassifier(max_depth=25, random_state=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[16]:


# print the accuracy and confusion matrix
print('Accuracy : ' + str(accuracy_score(y_test, y_pred)))
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m)

