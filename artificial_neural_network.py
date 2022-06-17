
# coding: utf-8

# # Artificial Neural Network

# ### Importing the libraries

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[2]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Importing the dataset

# In[3]:


dataset = pd.read_csv('https://github.com/anscholar/Galgotias-University/blob/main/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# In[4]:


print(X)


# In[5]:


print(y)


# ### Encoding categorical data

# Label Encoding the "Gender" column

# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])


# In[7]:


print(X)


# One Hot Encoding the "Geography" column

# In[8]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[9]:


print(X)


# ### Splitting the dataset into the Training set and Test set

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ### Feature Scaling

# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Part 2 - Building the ANN

# ### Initializing the ANN

# In[12]:


ann = tf.keras.models.Sequential()


# ### Adding the input layer and the first hidden layer

# In[13]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ### Adding the second hidden layer

# In[14]:


ann.add(tf.keras.layers.Dense(units=6, activation='relu'))


# ### Adding the output layer

# In[15]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Part 3 - Training the ANN

# ### Compiling the ANN

# In[16]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the ANN on the Training set

# In[17]:


ann.fit(X_train, y_train, batch_size = 32, epochs = 1)


# ## Part 4 - Making the predictions and evaluating the model

# ### Predicting the result of a single observation

# **Homework**
# 
# Use our ANN model to predict if the customer with the following informations will leave the bank: 
# 
# Geography: France
# 
# Credit Score: 600
# 
# Gender: Male
# 
# Age: 40 years old
# 
# Tenure: 3 years
# 
# Balance: \$ 60000
# 
# Number of Products: 2
# 
# Does this customer have a credit card ? Yes
# 
# Is this customer an Active Member: Yes
# 
# Estimated Salary: \$ 50000
# 
# So, should we say goodbye to that customer ?

# **Solution**

# In[18]:


print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# ### Predicting the Test set results

# In[19]:


y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ### Making the Confusion Matrix

# In[20]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

