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
url = "https://raw.githubusercontent.com/kmadden9/IntroToML/main/Housing.csv"
download = requests.get(url).content
# Reading the downloaded content and turning it into a pandas dataframe
housing = pd.read_csv(io.StringIO(download.decode('utf-8')))
housing.head() # To get first n rows from the dataset default value of n is 5
M = len(housing)


# In[2]:


#Now we have categorical variables with values like yes, no and furnished.
#We have to make these numerical to have them fit into our equations.

varlist = ['mainroad','guestroom','basement','hotwaterheating','airconditioning', 'prefarea']
furnitureList = ['furnishingstatus']

# Defining the map function
def binary_map(x):
    return x.map({'yes':1,'no':0})

def furniture_map(x):
    return x.map({'furnished':2, 'semi-furnished':1, 'unfurnished':0})

# Applying the functions to the housing list
housing[varlist] = housing[varlist].apply(binary_map)
housing[furnitureList] = housing[furnitureList].apply(furniture_map)

# Check the housing dataframe now
housing.head()


# In[3]:


y = housing.values[:, 0]  # The output of price is the first column
x1 = housing.values[:, 1]  # column 2 is 'area'
x2 = housing.values[:, 2]  # column 3 is 'bedrooms' 
x3 = housing.values[:, 3]  # column 4 is 'bathrooms'
x4 = housing.values[:, 4]  # column 5 is 'stories'
x5 = housing.values[:, 5]  # column 6 is 'mainroad' y/n
x6 = housing.values[:, 6]  # column 7 is 'guestroom' y/n
x7 = housing.values[:, 7]  # column 8 is 'basement' y/n
x8 = housing.values[:, 8]  # column 9 is 'hot water heating' y/n
x9 = housing.values[:, 9]  # column 10 is 'air conditioning' y/n
x10 = housing.values[:, 10]  # column 11 is 'parking spaces'
x11 = housing.values[:, 11]  # column 12 is 'prefarea' y/n
x12 = housing.values[:, 12]  # column 13 is 'furnishing status' furn/semi/none
m = len(y)  # Number of training examples
print('x1 = ', x1[: 5])
print('x2 = ', x2[: 5])
print('x3 = ', x3[: 5])
print('x4 = ', x4[: 5])
print('x5 = ', x5[: 5])
print('x6 = ', x6[: 5])
print('x7 = ', x7[: 5])
print('x8 = ', x8[: 5])
print('x9 = ', x9[: 5])
print('x10 = ', x10[: 5])
print('x11 = ', x11[: 5])
print('x12 = ', x12[: 5])
print('y = ', y[: 5])
print('m = ', m)


# In[4]:


plt.scatter(x1, y, color='red', marker= '+')
plt.scatter(x2, y, color='blue', marker= 'o')
plt.scatter(x3, y, color='yellow', marker= 'x')
plt.scatter(x4, y, color='orange', marker= '.')
plt.scatter(x5, y, color='purple', marker= '_')
plt.scatter(x6, y, color='cyan', marker= 's')
plt.scatter(x7, y, color='magenta', marker= 'd')
plt.scatter(x8, y, color='black', marker= '^')
plt.scatter(x9, y, color='pink', marker= 'v')
plt.scatter(x10, y, color='grey', marker= '>')
plt.scatter(x11, y, color='brown', marker= '<')
plt.scatter(x12, y, color='green', marker= 'p')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Housing Data')
plt.ylabel('Price ($)')
plt.title('Scatter plot price in relation to inputs')


# In[5]:


#Creating a Matrix with a column of ones
X_0 = np.ones((m,1))

#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows,
np.random.seed(0)
df_train,df_test=train_test_split(housing,train_size=0.7,test_size=0.3, random_state = 42, shuffle = True)
allVars = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement',
           'hotwaterheating','airconditioning','parking','prefarea','furnishingstatus','price']
df_Newtrain = df_train[allVars]
df_Newtest = df_test[allVars]
df_Newtrain.head()


# In[6]:


#from the scatter plot we can see that the inputs are not scaled for each other
#At the moment we won't scale them but this will be included in the next
#few questions. 
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler,StandardScaler
# define standard scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
df_Newtrain[allVars] = scaler.fit_transform(df_Newtrain[allVars])
df_Newtrain.head(20)


# In[7]:


df_Newtest[allVars] = scaler.fit_transform(df_Newtest[allVars])
df_Newtest.head(20)


# In[8]:


Y_Newtrain = df_Newtrain.pop('price')
X_Newtrain = df_Newtrain
Y_Newtest = df_Newtest.pop('price')
X_Newtest = df_Newtest


# In[9]:


X_Newtrain.head()


# In[10]:


Y_Newtrain.head()


# In[11]:


X_Newtest.head()


# In[12]:


Y_Newtest.head()


# In[13]:


theta = np.zeros(13)

# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final x matrix (feature matrix))
trainLength = len(X_Newtrain)
#Creating a new Matrix with a column of ones
X_0 = np.ones((trainLength,1))
xTrain = np.hstack((X_0, X_Newtrain))

# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final x matrix (feature matrix))
testLength = len(X_Newtest)
#Creating a new Matrix with a column of ones
X_0 = np.ones((testLength,1))
xTest = np.hstack((X_0, X_Newtest))
xTest[:5]


# In[14]:


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


# In[15]:


def compute_cost_test(xTest, Y_Newtest, theta):
    predictions = xTest.dot(theta)
    errors = np.subtract(predictions, Y_Newtest)
    sqrErrors = np.square(errors)
    J = 1 / (2 * testLength) * np.sum(sqrErrors)
    
    return J


# In[16]:


# Lets compute the cost for theta values
cost = compute_cost(xTrain, Y_Newtrain, theta)
print('The cost for given values of theta, 0-12 =', cost)


# In[17]:


def gradient_descent(xTrain, Y_Newtrain, theta, alpha, iterations, lambdaVal):
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
        reg_sum = (lambdaVal / trainLength) * sum(theta);
        theta = theta - sum_delta - reg_sum;
        cost_history[i] = compute_cost(xTrain, Y_Newtrain, theta)
        
    return theta, cost_history


# In[18]:


theta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
iterations = 1500;
alpha = 0.1;
lambdaVal = 1;


# In[19]:


theta, cost_history = gradient_descent(xTrain, Y_Newtrain, theta, alpha, iterations, lambdaVal)
print('Final value of theta =', theta)
print('cost_history =', cost_history)


# In[20]:


def cost_compute_test(xTest, Y_Newtest, theta, iterations):
    """"
    
    iterations: No of iterations. Scalar value.
    
    cost_history: Contains value of cost for each iteration. 1D array. Dimension(m x 1)
    """
    cost_history_test = np.zeros(iterations)
     
    for i in range(iterations):
        
        cost_history_test[i] = compute_cost_test(xTest, Y_Newtest, theta)
        
    return cost_history


# In[21]:


cost_history_test = cost_compute_test(xTest, Y_Newtest, theta, iterations)
print('cost history of test =', cost_history_test)


# In[22]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(xTrain[:,1],Y_Newtrain,color='red',marker='+',label='Training Data: Area')
plt.scatter(xTrain[:,2],Y_Newtrain,color='blue',marker='o',label='Training Data: Bedrooms')
plt.scatter(xTrain[:,3],Y_Newtrain,color='yellow',marker='x',label='Training Data: Bathrooms')
plt.scatter(xTrain[:,4],Y_Newtrain,color='orange',marker='.',label='Training Data: Stories')
plt.scatter(xTrain[:,5],Y_Newtrain,color='purple',marker='_',label='Training Data: Main Road')
plt.scatter(xTrain[:,6],Y_Newtrain,color='cyan',marker='s',label='Training Data: Guest Room')
plt.scatter(xTrain[:,7],Y_Newtrain,color='magenta',marker='d',label='Training Data: Basement')
plt.scatter(xTrain[:,8],Y_Newtrain,color='black',marker='^',label='Training Data: Hot Water Heater')
plt.scatter(xTrain[:,9],Y_Newtrain,color='pink',marker='v',label='Training Data: Air Conditioning')
plt.scatter(xTrain[:,10],Y_Newtrain,color='grey',marker='>',label='Training Data: Parking Station')
plt.scatter(xTrain[:,11],Y_Newtrain,color='brown',marker='<',label='Training Data: prefarea')
plt.scatter(xTrain[:,12],Y_Newtrain,color='green',marker='p',label='Training Data: Furnishing Status')


plt.plot(xTrain[:,1],xTrain.dot(theta),color='green',label='Linear Regression')

plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Scatter Plot of Input Data (alpha = 0.1) (lambda = 1)')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[23]:


plt.plot(range(1,iterations+1),cost_history,color='blue',label= 'loss convergence of training data')
plt.plot(range(1,iterations+1),cost_history_test,color='red',label = 'loss convergence of test data')
plt.legend()
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of Gradient Descent (alpha = 0.1) (lambda = 1)')


# In[24]:


#Creating a Matrix with a column of ones
X_0 = np.ones((m,1))

#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows,
np.random.seed(0)
df_train,df_test=train_test_split(housing,train_size=0.7,test_size=0.3, random_state = 42, shuffle = True)
allVars = ['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement',
           'hotwaterheating','airconditioning','parking','prefarea','furnishingstatus','price']
df_Newtrain = df_train[allVars]
df_Newtest = df_test[allVars]
df_Newtrain.head()


# In[25]:


#from the scatter plot we can see that the inputs are not scaled for each other
#At the moment we won't scale them but this will be included in the next
#few questions. 
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler,StandardScaler
# define standard scaler
scaler = StandardScaler()
#scaler = MinMaxScaler()
df_Newtrain[allVars] = scaler.fit_transform(df_Newtrain[allVars])
df_Newtrain.head(20)


# In[26]:


df_Newtest[allVars] = scaler.fit_transform(df_Newtest[allVars])
df_Newtest.head(20)


# In[27]:


Y_Newtrain = df_Newtrain.pop('price')
X_Newtrain = df_Newtrain
Y_Newtest = df_Newtest.pop('price')
X_Newtest = df_Newtest

theta = np.zeros(13)

# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final x matrix (feature matrix))
trainLength = len(X_Newtrain)
#Creating a new Matrix with a column of ones
X_0 = np.ones((trainLength,1))
xTrain = np.hstack((X_0, X_Newtrain))

# Lets use hstack() function from numpy to stack X_0 and X_1 horizontally (i.e. column
# This will be our final x matrix (feature matrix))
testLength = len(X_Newtest)
#Creating a new Matrix with a column of ones
X_0 = np.ones((testLength,1))
xTest = np.hstack((X_0, X_Newtest))
xTest[:5]


# In[28]:


# Lets compute the cost for theta values
cost = compute_cost(xTrain, Y_Newtrain, theta)
print('The cost for given values of theta, 0-12 =', cost)


# In[39]:


theta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
iterations = 1500;
alpha = 0.01;
lambdaVal = 1.5;


# In[40]:


theta, cost_history = gradient_descent(xTrain, Y_Newtrain, theta, alpha, iterations, lambdaVal)
print('Final value of theta =', theta)
print('cost_history =', cost_history)


# In[41]:


cost_history_test = cost_compute_test(xTest, Y_Newtest, theta, iterations)
print('cost history of test =', cost_history_test)


# In[44]:


# Since X is list of list (feature matrix) lets take values of column of index 1 only
plt.scatter(xTrain[:,1],Y_Newtrain,color='red',marker='+',label='Training Data: Area')
plt.scatter(xTrain[:,2],Y_Newtrain,color='blue',marker='o',label='Training Data: Bedrooms')
plt.scatter(xTrain[:,3],Y_Newtrain,color='yellow',marker='x',label='Training Data: Bathrooms')
plt.scatter(xTrain[:,4],Y_Newtrain,color='orange',marker='.',label='Training Data: Stories')
plt.scatter(xTrain[:,5],Y_Newtrain,color='purple',marker='_',label='Training Data: Main Road')
plt.scatter(xTrain[:,6],Y_Newtrain,color='cyan',marker='s',label='Training Data: Guest Room')
plt.scatter(xTrain[:,7],Y_Newtrain,color='magenta',marker='d',label='Training Data: Basement')
plt.scatter(xTrain[:,8],Y_Newtrain,color='black',marker='^',label='Training Data: Hot Water Heater')
plt.scatter(xTrain[:,9],Y_Newtrain,color='pink',marker='v',label='Training Data: Air Conditioning')
plt.scatter(xTrain[:,10],Y_Newtrain,color='grey',marker='>',label='Training Data: Parking Station')
plt.scatter(xTrain[:,11],Y_Newtrain,color='brown',marker='<',label='Training Data: prefarea')
plt.scatter(xTrain[:,12],Y_Newtrain,color='green',marker='p',label='Training Data: Furnishing Status')


plt.plot(xTrain[:,1],xTrain.dot(theta),color='green',label='Linear Regression')

plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Scatter Plot of Input Data (alpha = 0.01) (lambda = 1.5)')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[45]:


plt.plot(range(1,iterations+1),cost_history,color='blue',label= 'loss convergence of training data')
plt.plot(range(1,iterations+1),cost_history_test,color='red',label = 'loss convergence of test data')
plt.legend()
plt.rcParams["figure.figsize"]=(10,6)
plt.grid()
plt.xlabel('Number of Iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of Gradient Descent (alpha = 0.01) (lambda = 1.5)')


# In[ ]:




