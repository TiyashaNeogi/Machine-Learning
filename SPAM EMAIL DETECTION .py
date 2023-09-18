#!/usr/bin/env python
# coding: utf-8

# # SPAM EMAIL DETECTION WITH MACHINE LEARNING 

# We've all been the recipient of spam emails before. Spam mail, or junk mail, is a type of email that is sent to a massive number of users at one time, frequently containing cryptic messages, scams, or most dangerously, phishing content. In this Project, use Python to build an email spam detector. Then, use machine learning to train the spam detector to recognize and classify emails into spam and non-spam. Lets get started!

# #### import packages

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm 


# #### Load Dataset

# In[2]:


spam = pd.read_excel('C:\\Users\\Tiyasha Neogi\\Downloads\\CIPHERBYTE TECHNOLOGIES INTERNSHIP\\Spam Email Detection.xlsx')
spam


# <b>Text Processing</b>

# Cleaning the Raw Data

# In[3]:


spam.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1,inplace=True)


# In[4]:


spam.rename({'v1': 'spam','v2':'text'},axis=1,inplace=True)


# In[5]:


spam.columns


# Replace NaN values with empty strings

# In[6]:


spam = spam.where((pd.notnull(spam)),'')


# Lowering Case

# In[7]:


spam["text"] = spam["text"].str.lower()


# Removal of special characters

# In[8]:


import string
spam["text"] = spam["text"].str.translate(str.maketrans(dict.fromkeys(string.punctuation)))


# Removal of stop words

# In[9]:


from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
spam["text"] = [ 
    ' '.join([word for word in str(sentence).split() if word not in ENGLISH_STOP_WORDS]) 
    if isinstance(sentence, str) else sentence
    for sentence in spam["text"]
]


# <b>Label Encoding</b>

# In[10]:


# label spam mail as 0;  ham mail as 1;

spam.loc[spam['spam'] == 'spam', 'spam',] = 0
spam.loc[spam['spam'] == 'ham', 'spam',] = 1

spam['spam'] = spam['spam'].astype(int)


# In[11]:


# separating the data as texts and label

X = spam['text']
Y = spam['spam']


# <b> Train-Test Spliting </b>

# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# In[13]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# <b>Feature Extraction</b>

# Tokenizing the Cleaned Data

# In[14]:


X_train = X_train.fillna("")  # Replace NaN values with empty strings
X_test = X_test.fillna("")    # Replace NaN values with empty strings

from sklearn.feature_extraction.text import TfidfVectorizer

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)


# In[15]:


print(X_train_features)


# <b>Training the Model</b>
# <br/>Logistic Regression

# In[16]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# In[18]:


from sklearn.metrics import accuracy_score
# prediction on training data

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)


# In[19]:


print('Accuracy on training data : ', accuracy_on_training_data)


# In[20]:


# prediction on test data

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)


# In[21]:


print('Accuracy on test data : ', accuracy_on_test_data)


# <b>Performance Metrics</b>

# In[23]:


from sklearn.metrics import confusion_matrix,f1_score, precision_score,recall_score
cf_matrix =confusion_matrix(Y_test,prediction_on_test_data)
tn, fp, fn, tp = confusion_matrix(Y_test,prediction_on_test_data).ravel()
print("Precision: {:.2f}%".format(100 * precision_score(Y_test, prediction_on_test_data)))
print("Recall: {:.2f}%".format(100 * recall_score(Y_test, prediction_on_test_data)))
print("F1 Score: {:.2f}%".format(100 * f1_score(Y_test,prediction_on_test_data)))


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt
ax= plt.subplot()
#annot=True to annotate cells
sns.heatmap(cf_matrix, annot=True, ax = ax,cmap='Blues',fmt='');
# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['Not Spam', 'Spam']); ax.yaxis.set_ticklabels(['Not Spam', 'Spam']);


# In[ ]:




