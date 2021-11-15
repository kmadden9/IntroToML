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


breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)
breast_dataset.tail()


# In[10]:


#For question 1 we will consider all 30 input variables
X = breast_dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,
                    19,20,21,22,23,24,25,26,27,28,29]].values
Y = breast_dataset.iloc[:, 30].values

X[0:10]


# In[11]:


#Train test split
##from sklearn.model_selection import train_test_split
##X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, shuffle = True)


# In[12]:


#Now we’ll do feature scaling to scale our data between 0 and 1 to get better accuracy.

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() 
##X_train = sc_X.fit_transform(X_train) 
##X_test = sc_X.transform(X_test) 
X = sc_X.fit_transform(X)


# In[44]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents
            , columns = ['principal component 1', 'principal component 2', 'principal component 3')


# In[19]:




#finalDf = pd.concat([principalDf, breast_dataset[['label']]], axis = 1)
#finalDf


# In[34]:


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
xdata =  finalDf.iloc[:, 0].values
ydata = finalDf.iloc[:, 1].values
zdata = finalDf.iloc[:, 2].values
color = ['r', 'g', 'b']
ax.scatter3D(xdata, ydata, zdata, c='r');
ax.set_title('3 component PCA', fontsize = 20)
label = ['Benign', 'Malignant']

#for label, color in zip(label,colors):
   # indicesToKeep = finalDf['label'] == label
    #ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
           # , finalDf.loc[indicesToKeep, 'principal component 2']
           # , finalDf.loc[indicesToKeep, 'principal component 3']
          #  , c = color
          #  , s = 50)
#ax.legend(label)
#ax.grid()


# In[35]:


#change the label back to 0 and 1 for logistic regression
finalDf['label'].replace('Benign', 0,inplace=True)
finalDf['label'].replace('Malignant', 1,inplace=True)

finalDf


# In[36]:


#seperating the finalDF for the logitistic regression
X = finalDf.iloc[:, [0,1,2]].values
Y = finalDf.iloc[:, 3].values

X[0:10]


# In[37]:


#Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8,test_size=0.2, shuffle = True)


# In[38]:


#Import LogisticRegression from sklearn.linear_model 
#Make an instance classifier of the object LogisticRegression
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)


# In[39]:


Y_pred = classifier.predict(X_test) 
Y_pred[0:9] 


# In[40]:


#Using Confusion matrix we can get accuracy of our model. 
from sklearn.metrics import confusion_matrix 
cnf_matrix = confusion_matrix(Y_test, Y_pred) 
cnf_matrix


# In[41]:


#Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[42]:


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




