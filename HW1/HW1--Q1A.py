#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io
import seaborn as sns


# In[2]:


# Downloading the csv file from your GitHub account
url = "https://raw.githubusercontent.com/kmadden9/IntroToML/main/Housing.csv"
download = requests.get(url).content
# Reading the downloaded content and turning it into a pandas dataframe
housing = pd.read_csv(io.StringIO(download.decode('utf-8')))
housing.head() # To get first n rows from the dataset default value of n is 5
M = len(housing)
M


# In[3]:


y = housing.values[:, 0]  # The output of price is the first column
x1 = housing.values[:, 1]  # column 2 is 'area'
x2 = housing.values[:, 2]  # column 3 is 'bedrooms' 
x3 = housing.values[:, 3]  # column 4 is 'bathrooms'
x4 = housing.values[:, 4]  # column 5 is 'stories'
x5 = housing.values[:, 10]  # column 11 is 'parking spaces
m = len(y)  # Number of training examples
print('x1 = ', x1[: 5])
print('x2 = ', x2[: 5])
print('x3 = ', x3[: 5])
print('x4 = ', x4[: 5])
print('x5 = ', x5[: 5])
print('y = ', y[: 5])
print('m = ', m)


# In[4]:


plt.scatter(x1, y, color='red', marker= '+')
plt.scatter(x2, y, color='blue', marker= 'o')
plt.scatter(x3, y, color='green', marker= '.')
plt.scatter(x4, y, color='orange', marker= '_')
plt.scatter(x5, y, color='purple', marker= 's')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Data of Area, Bedrooms, Bathrooms, Stories, and Parking Spaces')
plt.ylabel('Price ($)')
plt.title('Scatter plot price in relation to inputs')


# In[5]:


#Creating a Matrix with a column of ones
X_0 = np.ones((m,1))
X_0[:5]


# In[6]:


#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows,
np.random.seed(0)
df_train,df_test=train_test_split(housing,train_size=0.7,test_size=0.3, random_state = 42, shuffle = True)
df_train.shape


# In[7]:


df_test.shape


# In[19]:


num_vars = ['area','bedrooms','bathrooms','stories','parking','price']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[20]:


df_Newtest.head()


# In[21]:


df_Newtrain.shape


# In[22]:


df_Newtest.shape


# In[23]:


#from the scatter plot we can see that the inputs are not scaled for each other
#At the moment we won't scale them but this will be included in the next
#few questions. 
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler,StandardScaler
# define standard scaler
#scaler = StandardScaler()
#scaler = MinMaxScaler()
#df_Newtrain[num_vars] = scaler.fit_transform(df_Newtrain[num_vars])
#df_Newtrain.head(20)


# In[24]:


Y_Newtrain = df_Newtrain.pop('price')
X_Newtrain = df_Newtrain
Y_Newtest = df_Newtest.pop('price')
X_Newtest = df_Newtest


# In[25]:


X_Newtrain.head()


# In[26]:


Y_Newtrain.head()


# In[27]:


X_Newtest.head()


# In[28]:


Y_Newtest.head()


# In[29]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final x matrix (feature matrix))
trainLength = len(X_Newtrain)
#Creating a new Matrix with a column of ones
X_0 = np.ones((trainLength,1))
xTrain = np.hstack((X_0, X_Newtrain))
xTrain[:5]


# In[30]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final x matrix (feature matrix))
testLength = len(X_Newtest)
#Creating a new Matrix with a column of ones
X_0 = np.ones((testLength,1))
X_0[:5]
xTest = np.hstack((X_0, X_Newtest))
xTest[:5]


# In[34]:


theta = np.zeros(6)


# In[47]:


def compute_cost(xTrain, Y_Newtrain, theta):
    """
    Compute cost for linear regression.
    
    Input Parameters
    ----------------
    x1 : 2D array where each row represent the training example and each column represent
         m= number of training examples
         n= number of freatures (including X_0 column of ones)
    y :  1D array of labels/target value for each training example. dimension(1 x m)
    
    theta : 1D array of fitting parameters or weights. Dimension (1 x n)
    
    Output Parameters
    -----------------
    J : Scalar value
    """
    predictions = xTrain.dot(theta)
    errors = np.subtract(predictions, Y_Newtrain)
    sqrErrors = np.square(errors)
    J = 1 / (2 * trainLength) * np.sum(sqrErrors)
    
    return J


# In[55]:


def compute_cost_test(xTest, Y_Newtest, theta):
    predictions = xTest.dot(theta)
    errors = np.subtract(predictions, Y_Newtest)
    sqrErrors = np.square(errors)
    J = 1 / (2 * testLength) * np.sum(sqrErrors)
    
    return J


# In[56]:


# Lets compute the cost for theta values
cost = compute_cost(xTrain, Y_Newtrain, theta)
print('The cost for given values of theta_0, theta_1, theta_2, theta_3, theta_4, and theta_5 =', cost)


# In[57]:


def gradient_descent(xTrain, Y_Newtrain, theta, alpha, iterations):
    """"
    Compute cost for linear regression.
    
    Input Parameters
    ----------------
    x1 : 2D array where each row represent the training example and each column represent
         m= number of training examples
         n= number of features (including X_0 column of ones)
    y : 1D array of labels/target value for each training example. dimension(m x 1)
    theta : 1D array of fitting parameters or weights. Dimension (1 x n)
    alpha : Learning rate. Scalar value
    iterations: No of iterations. Scalar value.
    
    Output Parameters
    -----------------
    theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n)
    alpha : Learning rate. Scalar value
    iterations: No of iterations. Scalar value.
    
    Output Parameters
    -----------------
    theta : Final Value. 1D array of fitting parameters or weights. Dimension (1 x n)
    cost_history: Contains value of cost for each iteration. 1D array. Dimension(m x 1)
    """
    cost_history = np.zeros(iterations)
     
    for i in range(iterations):
        predictions = xTrain.dot(theta)
        errors = np.subtract(predictions, Y_Newtrain)
        sum_delta = (alpha / trainLength) * xTrain.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compute_cost(xTrain, Y_Newtrain, theta)
        
    return theta, cost_history


# In[73]:


theta = [0., 0., 0., 0., 0., 0.]
iterations = 1500;
alpha = 0.000000001;


# In[74]:


theta, cost_history = gradient_descent(xTrain, Y_Newtrain, theta, alpha, iterations)
print('Final value of theta =', theta)
print('cost_history =', cost_history)


# In[75]:


def cost_compute_test(xTest, Y_Newtest, theta, iterations):
    """"
    
    iterations: No of iterations. Scalar value.
    
    cost_history: Contains value of cost for each iteration. 1D array. Dimension(m x 1)
    """
    cost_history_test = np.zeros(iterations)
     
    for i in range(iterations):
        
        cost_history_test[i] = compute_cost_test(xTest, Y_Newtest, theta)
        
    return cost_history


# In[76]:


cost_history_test = cost_compute_test(xTest, Y_Newtest, theta, iterations)
print('cost history of test =', cost_history_test)


# In[77]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(xTrain[:,1],Y_Newtrain,color='red',marker='+',label='Training Data: Input 1')
plt.scatter(xTrain[:,2],Y_Newtrain,color='blue',marker='o',label='Training Data: Input 2')
plt.scatter(xTrain[:,3],Y_Newtrain,color='yellow',marker='x',label='Training Data: Input 3')
plt.scatter(xTrain[:,4],Y_Newtrain,color='orange',marker='x',label='Training Data: Input 4')
plt.scatter(xTrain[:,5],Y_Newtrain,color='purple',marker='x',label='Training Data: Input 5')

plt.plot(xTrain[:,1],xTrain.dot(theta),color='green',label='Linear Regression')

plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Scatter Plot of Input Data (alpha = 0.000000001)')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[78]:


plt.plot(range(1,iterations+1),cost_history,color='blue',label= 'loss convergence of training data')
plt.plot(range(1,iterations+1),cost_history_test,color='red',label = 'loss convergence of test data')
plt.legend()
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of Gradient Descent (alpha = 0.000000001)')


# In[ ]:




