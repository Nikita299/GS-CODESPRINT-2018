
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np


# In[2]:


from sklearn.model_selection import train_test_split
import operator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
import numpy
import keras
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model import LogisticRegression


# In[3]:


data = pd.read_csv('train.csv')



# In[7]:


train = data
train_x=train.drop('popularity',axis=1)
train_y=train['popularity']


# In[31]:


test = pd.read_csv('test.csv')


# In[35]:


#exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=8, max_features=0.8, min_samples_leaf=3, min_samples_split=11, n_estimators=100, subsample=0.85)
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.6, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
)
exported_pipeline.fit(train_x, train_y)
results = exported_pipeline.predict(test)




# In[37]:

results = pd.Series(results)
results.to_csv("prediction.csv", sep=',',encoding='utf-8', index=False)





