#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib


# In[2]:


from data_preprocessing import (
    text_data_cleaning,
    tfidf_feature_fit,
    tfidf_features_transform
)


# In[3]:


from data_splitting import prepare_data
from model_building import (fit_and_evaluate_model,get_feature_importance)


# In[4]:


data=pd.read_csv('preprocessed_data.csv')


# In[5]:


data.isnull().sum()


# In[6]:


data['review'] = data['review'].fillna('')


# In[7]:


# data.to_csv("preprocessed_data1.csv", index=False)


# In[8]:


x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(data)


tfidf, X_train_tfidf = tfidf_feature_fit(x_train)

X_val_tfidf = tfidf_features_transform(tfidf, x_val)

X_test_tfidf = tfidf_features_transform(tfidf, x_test)


model = fit_and_evaluate_model(
    X_train_tfidf,
    X_val_tfidf,
    y_train,
    y_val
)


joblib.dump(model, 'sentiment_model.pkl')


joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model saved successfully.")


importance = get_feature_importance(model, tfidf)

print(importance.head(20))

