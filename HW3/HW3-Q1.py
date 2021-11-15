#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer


# In[2]:


breast = load_breast_cancer()
breast_data = breast.data
breast_data.shape


# In[3]:


breast_input = pd.DataFrame(breast_data)
breast_input.head()


# In[4]:


breast_labels = breast.target
breast_labels.shape


# In[5]:


labels = np.reshape(breast_labels,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)
final_breast_data.shape


# In[6]:


breast_dataset = pd.DataFrame(final_breast_data)


# In[7]:


features = breast.feature_names
features


# In[8]:


features_labels = np.append(features,'label')
breast_dataset.columns = features_labels
breast_dataset.head()


# In[9]:


#breast_dataset['label'].replace(0, 'Benign',inplace=True)
#breast_dataset['label'].replace(1, 'Malignant',inplace=True)
breast_dataset.tail()


# In[10]:


#For question 1 we will consider all 30 input variables
X = breast_dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
                    19,20,21,22,23,24,25,26,27,28,29]].values
Y = breast_dataset.iloc[:, 30].values

X[0:10]


# In[11]:


#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, shuffle = True)


# In[12]:


#Now we’ll do feature scaling to scale our data between 0 and 1 to get better accuracy.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test) 


# In[13]:


#Import LogisticRegression from sklearn.linear_model 
#Make an instance classifier of the object LogisticRegression
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)


# In[14]:


Y_pred = classifier.predict(X_test) 
Y_pred[0:9] 


# In[15]:


#Using Confusion matrix we can get accuracy of our model. 
from sklearn.metrics import confusion_matrix 
cnf_matrix = confusion_matrix(Y_test, Y_pred) 
cnf_matrix


# In[16]:


#Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[17]:


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


# In[ ]:




