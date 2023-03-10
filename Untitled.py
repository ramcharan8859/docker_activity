#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install klib')


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import klib 
import warnings
warnings.filterwarnings("ignore")


# In[7]:


data=pd.read_csv("BankNote_Authentication.csv")
data


# In[9]:


data.head()


# In[10]:


data.tail()


# In[13]:


data.dtypes


# # EDA(exploratory data analysis)

# In[14]:


data.describe()


# In[15]:


data.info()


# # checking the null values

# In[16]:


# another method to find the missing values
[features for features in data.columns if data[features].isnull().sum()>0]
# here their is no missing values in the given  data 


# In[17]:


data.nunique()


# # visualization 

# In[19]:


#Density Plot
# density plot represents the probability of a data point falling within a certain range
data.plot(kind='density', subplots=True, layout=(5,5), sharex=False,figsize=(15,14))
plt.show()


# In[22]:


data.value_counts("class")


# In[20]:


plt.figure(figsize = (10,10))
sns.heatmap(data.corr(), annot = True,cmap="Blues",cbar=True)


# # Data preprocessing

# # type convertion

# In[32]:


num_attr = data.select_dtypes(['float64']).columns
num_attr
data[num_attr]=data[num_attr].astype('float64')
num_attr


# In[29]:


cat_attr = data.select_dtypes('int64').columns
data[cat_attr]=data[cat_attr].astype('category')
cat_attr


# In[33]:


data.dtypes


# In[35]:


X=data.drop(['class'],axis=1)
y=data['class']


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123,stratify=y)


# In[37]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score,recall_score,precision_score


# In[41]:


logistic_model = LogisticRegression()

logistic_model.fit(X_train,y_train)


# In[42]:


train_preds = logistic_model.predict(X_train)
train_preds_prob=logistic_model.predict_proba(X_train)[:,1]
test_preds = logistic_model.predict(X_test)
test_preds_prob=logistic_model.predict_proba(X_test)[:,1]


# In[43]:


train_preds


# In[44]:


logistic_model.coef_


# In[45]:


confusion_matrix(y_train,train_preds)


# In[46]:


train_accuracy_1= accuracy_score(y_train,train_preds)
train_recall_1= recall_score(y_train,train_preds)
train_precision_1= precision_score(y_train,train_preds)

test_accuracy_1= accuracy_score(y_test,test_preds)
test_recall_1= recall_score(y_test,test_preds)
test_precision_1= precision_score(y_test,test_preds)


# In[47]:


print(train_accuracy_1)
print(train_recall_1)
print(train_precision_1)

print(test_accuracy_1)
print(test_recall_1)
print(test_precision_1)


# In[48]:


#Classification report
print(classification_report(y_train,train_preds))


# In[49]:


print(classification_report(y_test,test_preds))


# In[50]:


from sklearn.ensemble import RandomForestClassifier
import pickle


# In[51]:


classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)


# In[52]:


y_pred=classifier.predict(X_test)


# In[53]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)


# In[54]:


score


# In[55]:


pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[56]:


classifier.predict([[2,3,4,1]])


# In[ ]:




