#!/usr/bin/env python
# coding: utf-8

# In[3]:


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


# In[4]:


diabetes.head() 


# In[5]:


X = diabetes.iloc[:, [0,1,2,3,4,5,6,7]].values
Y = diabetes.iloc[:, 8].values


# In[6]:


X[0:10] 


# In[7]:


#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, shuffle = True)


# In[8]:


#Now we’ll do feature scaling to scale our data between 0 and 1 to get better accuracy.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test) 


# In[9]:


#Import LogisticRegression from sklearn.linear_model 
#Make an instance classifier of the object LogisticRegression
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)


# In[10]:


Y_pred = classifier.predict(X_test) 


# In[11]:


Y_pred[0:9] 


# In[12]:


#Using Confusion matrix we can get accuracy of our model. 
from sklearn.metrics import confusion_matrix 
cnf_matrix = confusion_matrix(Y_test, Y_pred) 
cnf_matrix


# In[13]:


#Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[14]:


#Let's visualize the results of the model in the form of a co#nfusion matrix using matp
#Here, you will visualize the confusion matrix using Heatmap.
import seaborn as sns
class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Diabetes Diagnosis', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[15]:


#Now let's try Naive Gaussian Bays
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)


# In[16]:


Y2_pred = classifier.predict(X_test)


# In[17]:


Y2_pred


# In[18]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test, Y2_pred)
ac = accuracy_score(Y_test, Y2_pred)


# In[19]:


cm


# In[20]:


ac


# In[21]:


# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix for Diabetes Diagnosis', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[22]:


print("Accuracy:",metrics.accuracy_score(Y_test, Y2_pred))
print("Precision:",metrics.precision_score(Y_test, Y2_pred))
print("Recall:",metrics.recall_score(Y_test, Y2_pred))


# In[35]:





# In[36]:





# In[ ]:





# In[ ]:




