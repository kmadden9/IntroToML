#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io


# In[2]:


# Downloading the csv file from your GitHub account
url = "https://raw.githubusercontent.com/kmadden9/IntroToML/main/D3.csv"
download = requests.get(url).content
# Reading the downloaded content and turning it into a pandas dataframe
df = pd.read_csv(io.StringIO(download.decode('utf-8')))
df.head() # To get first n rows from the dataset default value of n is 5
M = len(df)
M


# In[4]:


x = df.values[:, 2]  # get input values from third column
y = df.values[:, 3]  # get output values from fourth column
m = len(y)  # Number of training examples
print('x = ', x[: 5])
print('y = ', y[: 5])
print('m = ', m)


# In[5]:


plt.scatter(x, y, color='red', marker= '+')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Input 1')
plt.ylabel('Output')
plt.title('Scatter plot of training data')


# In[6]:


#Lets create a matrix with single column of ones
X_0 = np.ones((m,1))
X_0[:5]


# In[7]:


# Using reshape function convert x 1D array to 2D array of dimension 97x1
X_1 = x.reshape(m, 1)
X_1[:10]


# In[8]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final x matrix (feature matrix))
x = np.hstack((X_0, X_1))
x[:5]


# In[9]:


theta = np.zeros(2)
theta


# In[10]:


def compute_cost(x, y, theta):
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
    predictions = x.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    
    return J


# In[11]:


# Lets compute the cost for theta values
cost = compute_cost(x, y, theta)
print('The cost for given values of theta_0 and theta_1 =', cost)


# In[12]:


def gradient_descent(x, y, theta, alpha, iterations):
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
        predictions = x.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * x.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compute_cost(x, y, theta)
        
    return theta, cost_history


# In[13]:


theta = [0., 0.]
iterations = 1500;
alpha = 0.01;


# In[14]:


theta, cost_history = gradient_descent(x, y, theta, alpha, iterations)
print('Final value of theta =', theta)
print('cost_history =', cost_history)


# In[18]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(x[:,1],y,color='red',marker='+',label='Training Data')
plt.plot(x[:,1],x.dot(theta),color='green',label='Linear Regression')

plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Scatter Plot of Column 3 Input Data')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[17]:


plt.plot(range(1,iterations+1),cost_history,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of Gradient Descent')


# In[ ]:




