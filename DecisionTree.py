
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[1]:


hawaF = './data/features-hawa.csv'
hawaL = './data/onehotLabels-hawa.csv'
myF = './data/featuredata.csv'
myL = './data/onehotdata.csv'


# In[3]:


X = np.loadtxt(myF, delimiter = ",")
y = np.loadtxt(myL, delimiter = ",")


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=10)
clf = DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=10)
res = clf.fit(X_train, y_train)

print('Finish DT training')

predicttest = res.predict(X_test)


# In[11]:


acc = metrics.accuracy_score(y_test, predicttest)
print(acc)

