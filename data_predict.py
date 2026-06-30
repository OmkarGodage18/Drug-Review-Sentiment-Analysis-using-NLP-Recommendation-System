#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib

from data_preprocessing import (
    text_data_cleaning,
    tfidf_features_transform
)

model = joblib.load("sentiment_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")


data = pd.read_csv("drug_reviews_sentiment_analysis.csv")

test_data=pd.read_csv("test_data.csv")
test_data = text_data_cleaning(test_data)

# Fill missing reviews if any
test_data['review'] = test_data['review'].fillna('')

# =========================
# TFIDF TRANSFORM
# =========================

test_tfidf = tfidf_features_transform(
    tfidf,
    test_data
)

# =========================
# PREDICTION
# =========================

predictions = model.predict(test_tfidf)

probabilities = model.predict_proba(test_tfidf)


test_data['predicted_sentiment'] = predictions

test_data['negative_probability'] = probabilities[:, 0]

test_data['positive_probability'] = probabilities[:, 1]


print(
    test_data[
        [
            'review_original',
            'predicted_sentiment',
            'negative_probability',
            'positive_probability'
        ]
    ].head(10)
)

test_data.to_csv(
    "prediction_results.csv",
    index=False
)

print("\nPrediction file saved successfully.")


# In[ ]:




