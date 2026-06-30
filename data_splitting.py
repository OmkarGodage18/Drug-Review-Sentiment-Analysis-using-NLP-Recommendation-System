#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import (text_data_cleaning,tfidf_feature_fit,tfidf_features_transform)


# In[3]:


data= pd.read_csv('drug_reviews_sentiment_analysis.csv')


# In[4]:


data=text_data_cleaning(data)


# In[5]:


data.to_csv("preprocessed_data.csv", index=False)


# In[6]:


from sklearn.model_selection import train_test_split

def prepare_data(data):

    # Feature and target
    x = data[['review']]

    y = data['sentiment']

    # Train split
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    # Validation and test split
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.3,
        random_state=42,
        stratify=y_temp
    )

    # Save train
    train_df = x_train.copy()
    train_df['sentiment'] = y_train
    train_df.to_csv('train_data.csv', index=False)

    # Save validation
    val_df = x_val.copy()
    val_df['sentiment'] = y_val
    val_df.to_csv('val_data.csv', index=False)

    # Save test
    test_df = x_test.copy()
    test_df['sentiment'] = y_test
    test_df.to_csv('test_data.csv', index=False)

    return x_train, x_val, x_test, y_train, y_val, y_test


# In[7]:


get_ipython().system('jupyter nbconvert --to script data_splitting.ipynb')


# In[8]:


data.head(10)


# In[ ]:




