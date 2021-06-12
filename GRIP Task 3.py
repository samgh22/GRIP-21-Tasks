#!/usr/bin/env python
# coding: utf-8

# # GRIP Task 3 : Exploratory Data Analysis - Retail
# 
# **Importing the necessary Libraries**

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Loading the Dataset**

# In[5]:


df = pd.read_csv("SampleSuperstore.csv")
df.head()


# **Finding the shape of the dataset**

# In[6]:


print(df.shape)


# **Description of the Dataset**

# In[7]:


df.describe(include = 'all')


# In[8]:


print(df.isna().sum())


# In[9]:


df.drop_duplicates(keep = 'first' , inplace = True)


# In[10]:


corr = df.corr()
sns.heatmap(corr, annot = True)


# In[11]:


sns.countplot(df['Segment'])
plt.show()


# In[23]:


plt.figure(figsize = (16,9))
plt.bar('State', 'Sales',data = df)
plt.xticks(rotation = 90)
plt.show()


# In[26]:


plt.figure(figsize = (10,10))
plt.bar('Sub-Category', 'Category', data = df)
plt.xlabel("Sub-Category")
plt.ylabel("Category")
plt.title("Category vs Sub-Category")
plt.xticks(rotation = 90)
plt.show()


# In[28]:


plt.figure(figsize = (16,8))
sns.countplot(df['State'])
plt.xticks(rotation=90)
plt.show()


# In[29]:


df.hist(figsize=(10,10), bins = 50)
plt.show()


# In[ ]:




