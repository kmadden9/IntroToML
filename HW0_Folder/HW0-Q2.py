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


x1 = df.values[:, 0]  # get input values from first column
x2 = df.values[:, 1]  # get input values from second column
x3 = df.values[:, 2]  # get input values from third column
y = df.values[:, 3]  # get output values from fourth column
m = len(y)  # Number of training examples
print('x1 = ', x1[: 5])
print('x2 = ', x2[: 5])
print('x3 = ', x3[: 5])
print('y = ', y[: 5])
print('m = ', m)


# In[6]:


plt.scatter(x1, y, color='red', marker= '+')
plt.scatter(x2, y, color='blue', marker= 'o')
plt.scatter(x3, y, color='green', marker= 'x')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Inputs 1, 2, and 3')
plt.ylabel('Output')
plt.title('Scatter plot of training data')


# In[7]:


#Lets create a matrix with single column of ones
X_0 = np.ones((m,1))
X_0[:5]


# In[8]:


# Using reshape function convert x 1D array to 2D array of dimension 100x1
X_1 = x1.reshape(m, 1)
X_1[:10]
X_2 = x2.reshape(m, 1)
X_2[:10]
X_3 = x3.reshape(m, 1)
X_3[:10]


# In[9]:


# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final x matrix (feature matrix))
x = np.hstack((X_0, X_1, X_2, X_3))
x[:5]


# In[10]:


theta = np.zeros(4)
theta


# In[11]:


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


# In[13]:


# Lets compute the cost for theta values
cost = compute_cost(x, y, theta)
print('The cost for given values of theta_0, theta_1, theta_2, and theta_3 =', cost)


# In[14]:


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


# In[55]:


theta = [0., 0., 0., 0.]
iterations = 1500;
alpha = 0.1;


# In[56]:


theta, cost_history = gradient_descent(x, y, theta, alpha, iterations)
print('Final value of theta =', theta)
print('cost_history =', cost_history)


# In[59]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(x[:,1],y,color='red',marker='+',label='Training Data: Input 1')
plt.scatter(x[:,2],y,color='blue',marker='o',label='Training Data: Input 2')
plt.scatter(x[:,3],y,color='green',marker='x',label='Training Data: Input 3')
plt.plot(x[:,1],x.dot(theta),color='green',label='Linear Regression')

plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Scatter Plot of Input Data (alpha = 0.1)')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[60]:


plt.plot(range(1,iterations+1),cost_history,color='blue')
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of Gradient Descent (alpha = 1)')


# In[ ]:




