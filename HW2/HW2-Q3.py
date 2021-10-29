#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
import seaborn as sns

# Downloading the csv file from your GitHub account
url = "https://raw.githubusercontent.com/kmadden9/IntroToML/Homework/HW2/diabetes.csv"
download = requests.get(url).content
# Reading the downloaded content and turning it into a pandas dataframe
diabetes = pd.read_csv(io.StringIO(download.decode('utf-8')))
diabetes.head() # To get first n rows from the dataset default value of n is 5
M = len(diabetes)


# In[2]:


diabetes.head() 


# In[3]:


X = diabetes.iloc[:, [0,1,2,3,4,5,6,7]].values
Y = diabetes.iloc[:, 8].values


# In[4]:


X[0:10] 


# In[5]:


#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, shuffle = True)


# In[15]:


#Now we’ll do feature scaling to scale our data between 0 and 1 to get better accuracy.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test) 
X = sc_X.fit_transform(X)


# In[37]:


#Import LogisticRegression from sklearn.linear_model 
#Make an instance classifier of the object LogisticRegression
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test) 


# In[38]:


#Using Confusion matrix and other imports we can get accuracy of our model. 
from sklearn.metrics import confusion_matrix 
from sklearn import metrics
import seaborn as sns
#importing k-fold validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score


# In[41]:


kf = KFold(n_splits=5, random_state=0, shuffle=True)
kf


# In[42]:


for train_index, test_index in kf.split(X):
    print(train_index, test_index)


# In[43]:


cross_val_score(classifier, X, Y)


# In[44]:


kf = KFold(n_splits=10, random_state=0, shuffle=True)
kf


# In[45]:


for train_index, test_index in kf.split(X):
    print(train_index, test_index)


# In[47]:


cross_val_score(classifier, X, Y, cv=10)


# In[ ]:




