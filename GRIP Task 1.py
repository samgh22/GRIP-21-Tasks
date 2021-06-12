#!/usr/bin/env python
# coding: utf-8

# # GRIP Task 1 : Prediction using Supervised ML 
# 
# **We have to predict the percentage of a student based on the number of study hours**

# **Importing the required Datasets and Libraries**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("The data has been imported")
df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# # Data Visualization
# 
# Plotting the data in order to get a clear idea about the relation of the **number of hours studied** and the **scores obtained** 

# In[6]:


df.plot(x = 'Hours', y = 'Scores' , style = 'o')
plt.xlabel('Number of Hours Studied')
plt.ylabel('Percentage of Scores Obtained')
plt.title('Hours vs Percentage')
plt.show()


# **From the scatter plot it has been observed that there is a clear positive linear relation between the Number of Hours Studied and the Percentage of Scores Obtained**

# In[8]:


plt.boxplot(df)


# # Linear Regression
# 
# **Now we separate the train and test data**

# In[9]:


X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.80, test_size = 0.2, random_state = 42)


# # Model Training

# In[10]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)


# In[11]:


print(X_test)


# # Prediction

# In[15]:


y_pred = lm.predict(X_test)
print(y_pred)


# # Comparison of Actual of Predicted Values

# In[19]:


df1 = pd.DataFrame({'Actual' : Y_test, 'Predicted' : y_pred})
df1.head()


# # Plotting the Regression Line

# In[23]:


regression_line = lm.coef_*X + lm.intercept_
plt.scatter(X,Y)
plt.plot(X,regression_line)
plt.title('Hours vs Percentage')
plt.xlabel('Number of Hours Studied')
plt.ylabel('Percentage of Scores Obtained')
plt.show()


# In[24]:


print('Test Score')
print(lm.score(X_test,Y_test))
print('Train Score')
print(lm.score(X_train, Y_train))


# # Prediction of the Score of a Student who Studies 9.25 hrs/day

# In[25]:


result = lm.predict([[9.25]])
print('Score of the Student who studies 9.25 hrs/day ', result)


# # Evaluating Model

# In[27]:


from sklearn import metrics
print('Mean Absolute Error is ', metrics.mean_absolute_error(Y_test, y_pred))
print('Mean Squared Error is ' , metrics.mean_squared_error(Y_test, y_pred))

