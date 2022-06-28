#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_excel('iris.xls')


# In[3]:


data.head()


# # handling missing values

# In[4]:


data.isna().sum()


# In[5]:


data.dtypes


# In[6]:


freqgraph = data.select_dtypes(include=['float'])     
freqgraph.hist(figsize=(20,15))
plt.show()


# In[7]:


data['PL'] = data['PL'].fillna(data['PL'].median())


# In[8]:


data.isna().sum()


# In[9]:


for i in ['SL', 'SW']:
    data[i] = data[i].fillna(data[i].mean())


# In[10]:


data.isna().sum()


# # All missing values are filled
# #outlier detection

# In[11]:


for i in ['PL', 'PW', 'SL', 'SW']:
    plt.figure()
    plt.boxplot(data[i])
    plt.title(i)


# In[12]:


Q1 = np.percentile(data['SW'], 25, interpolation = 'midpoint')  
Q2 = np.percentile(data['SW'], 50, interpolation = 'midpoint')  
Q3 = np.percentile(data['SW'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1
low_lim = Q1 - 1.5 * IQR 
up_lim = Q3 + 1.5 * IQR


# In[13]:


print(low_lim)
print(up_lim)


# In[14]:


ind1 = data['SW']<low_lim
data.loc[ind1].index


# In[15]:


ind2 = data['SW']>up_lim
data.loc[ind2].index


# In[16]:


plt.boxplot(data['SW'])


# # outliers are eliminated#label encoding
# 

# In[17]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Classification'] = le.fit_transform(data['Classification'])


# In[18]:


data.head()


# In[19]:


data['Classification'].unique()


# # splitting

# In[20]:


X = data.drop(['Classification'], axis=1)
y = data['Classification']


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)


# In[22]:


X_train.shape


# In[23]:


X_test.shape


# # Logistics regression

# In[24]:


from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
logit_model.fit(X_train,y_train)
y_pred = logit_model.predict(X_test)


# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# # KNN

# In[30]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[31]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# In[32]:


y_pred = classifier.predict(X_test)


# In[33]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[ ]:





# # SVM

# In[36]:


from sklearn.svm import SVC
sm = SVC()
sm.fit(X_train, y_train)
y_pred = sm.predict(X_test)


# In[37]:


print(classification_report(y_test, y_pred))


# # Decision tree

# In[38]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


# In[39]:


print(classification_report(y_test, y_pred))


# # Random Forest

# In[40]:


from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier()
rc.fit(X_train, y_train)
y_pred = rc.predict(X_test)


# In[41]:


print(classification_report(y_test, y_pred))


# # Scaling

# In[42]:


X.describe()


# In[43]:


from sklearn.preprocessing import StandardScaler
standardisation = StandardScaler()
X_train = standardisation.fit_transform(X_train)
X_test = standardisation.fit_transform(X_test)


# In[44]:


logit_model = LogisticRegression()
logit_model.fit(X_train,y_train)
y_pred = logit_model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[46]:


classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))


# In[47]:


sm = SVC()
sm.fit(X_train, y_train)
y_pred = sm.predict(X_test)
print(classification_report(y_test, y_pred))


# In[49]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(classification_report(y_test, y_pred))


# In[50]:


rc = RandomForestClassifier()
rc.fit(X_train, y_train)
y_pred = rc.predict(X_test)
print(classification_report(y_test, y_pred))


# In[ ]:




