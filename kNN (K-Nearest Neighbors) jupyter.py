#!/usr/bin/env python
# coding: utf-8

# ## Importing Iris dataset 

# In[157]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# ![image.png](attachment:image.png)

# In[158]:


iris.feature_names


# In[159]:


iris.target_names


# ## Creating DataFrame

# In[160]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[161]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# ## Add Target Column

# In[162]:


df['target'] = iris.target
df.head()


# ## Filter Data for a Specific Target

# In[128]:


df[df.target==1].head()


# In[129]:


df[df.target==2].head()


# Add Flower Name Column

# In[163]:


df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()


# ## View Rows from Index 45 to 55

# In[164]:


df[45:55]


# ## Split Data by Target Values

# In[165]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# ## Visualize Sepal Dimensions

# In[166]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[134]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


# ## Visualize Petal Dimensions

# In[135]:


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')


# ## Prepare Data for Modeling

# In[167]:


from sklearn.model_selection import train_test_split


# In[168]:


X = df.drop(['target','flower_name'], axis='columns')
y = df.target


# ## Split the Data

# In[169]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Parameters:
# test_size=0.2: 20% of the data is reserved for testing.
# random_state=1: Ensures reproducibility.
# 
# 
# 

# In[170]:


len(X_train)


# len(X_train): Returns the number of training samples (120 in this case)

# In[171]:


len(X_test)


# len(X_test): Returns the number of testing samples (30 in this case)

# ## Initialize the kNN Classifier

# In[180]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)


# ## Train the Model

# In[181]:


knn.fit(X_train, y_train)


# In[182]:


knn.score(X_test, y_test)


# ## Make a Prediction

# In[183]:


input_data = pd.DataFrame([[4.8, 3.0, 1.5, 0.3]], columns=X.columns)

knn.predict(input_data)


# ## Generate the Confusion Matrix

# In[184]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm


# ## Visualize the Confusion Matrix

# In[185]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# ## Display the Classification Report

# In[186]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[ ]:




