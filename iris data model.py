#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

# Iris flower has three species; 
# <li>setosa</li>
# <li>versicolor</li>
# <li>virginica</li> 
# which differs according to their measurements.Here our task is to train a machine learning model that can learn from the measurements of the iris species and classify them.

# #### import packages

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# #### Load Dataset

# In[6]:


df = pd.read_excel("Downloads\\Iris Flower.xlsx")
df


# In[9]:


df.info()


# #### Missing Value checking

# In[8]:


df.isnull().sum()


# Drop Unnecessary Columns

# In[10]:


df.drop(["Id"],axis=1,inplace=True)


# In[11]:


df.columns


# In[13]:


df.shape


# In[15]:


df.describe()


# Class Destribution of flowers' species

# In[17]:


df.groupby("Species").count()


# #### Visualization

# BOX & WHISKER PLOTS

# In[28]:


df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, 
        title="Box and Whisker plot for each attribute")


# In[19]:


#Visualize the whole dataset
sns.pairplot(df,hue = "Species")


# #### Data Modeling

# Train - Test Spliting

# In[36]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)


# In[42]:


print(X_train.shape,y_train.shape)


# In[43]:


print(X_test.shape,y_test.shape)


# Model Building

# In[45]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[46]:


svc =  SVC(gamma="auto")
KFold=KFold(n_splits=10)
cv_res=cross_val_score(svc,X_train,y_train,cv=KFold,scoring="accuracy")


# In[52]:


print("accuracy = "+str(cv_res.mean()), ", std = "+str(cv_res.std()))


# In[53]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[54]:


svc.fit(X_train,y_train)


# In[55]:


pred=svc.predict(X_test)


# In[57]:


print("Accuracy =" ,accuracy_score(y_test,pred))


# In[59]:


print("Confusion Matrix")
print(confusion_matrix(y_test,pred))


# In[60]:


print("Classification Report")
print(classification_report(y_test,pred))


# In[ ]:




